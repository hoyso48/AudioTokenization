import time
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import wandb
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from AudioTokenization.BigCodec_NNX.codec_module import CodecModule


class CodecTrainer:
    """
    Minimal Flax NNX trainer skeleton for CodecModule.

    Responsibilities in this skeleton:
    - Construct model from Hydra cfg
    - Run forward pass and compute losses (no updates yet)
    - Provide a fit() loop that prints losses for quick smoke testing

    Next phases will add:
    - Dual Optax optimizers (generator/discriminator) and updates
    - Sharding (TPUv4-8), Orbax checkpointing, W&B logging, metrics, etc.
    """

    def __init__(self, cfg: Any, rng_seed: int = 0):
        self.cfg = cfg
        self.global_step = 0

        # RNGs: provide default and 'dropout' streams to satisfy nnx.Dropout
        self.rngs = nnx.Rngs(rng_seed, params=rng_seed, dropout=rng_seed + 1)

        # Model
        self.model = CodecModule(cfg=self.cfg, rngs=self.rngs)

        # Optimizers (initialized lazily on first step)
        self.gen_tx: Optional[optax.GradientTransformation] = None
        self.disc_tx: Optional[optax.GradientTransformation] = None
        self.gen_opt_state: Optional[optax.OptState] = None
        self.disc_opt_state: Optional[optax.OptState] = None

        # W&B (optional)
        self._wandb_enabled = False
        try:
            project = getattr(self.cfg, 'project', 'Audio-Tokenizer')
            name = getattr(self.cfg, 'name', 'nnx-run')
            save_dir = getattr(self.cfg, 'log_dir', None)
            run_id = getattr(self.cfg, 'wandb_id', None)
            wandb.init(project=project, name=name, dir=save_dir, id=run_id, resume='allow')
            self._wandb_enabled = True
        except Exception:
            self._wandb_enabled = False

        # Sharded compute cache
        self._mesh: Optional[Mesh] = None
        self._batch_sharding: Optional[NamedSharding] = None
        self._compute_grads_sharded = None

    # -------------------- Helpers for param/grads partition --------------------
    @staticmethod
    def _to_pure(state: nnx.State) -> Dict:
        return state.to_pure_dict() if hasattr(state, 'to_pure_dict') else state

    @staticmethod
    def _state_arrays(pure_state: Dict) -> Dict:
        return jax.tree.map(lambda v: getattr(v, 'value', v), pure_state)

    @staticmethod
    def _filter_top_keys(pure_tree: Dict, include: set) -> Dict:
        return {k: v for k, v in pure_tree.items() if k in include}

    @staticmethod
    def _exclude_top_keys(pure_tree: Dict, exclude: set) -> Dict:
        return {k: v for k, v in pure_tree.items() if k not in exclude}

    def _split_params(self) -> Tuple[Dict, Dict]:
        """Split model params into generator vs discriminator top-level subtrees."""
        params_state = nnx.state(self.model, nnx.Param)
        params_pure = self._to_pure(params_state)
        disc_keys = {'discriminator', 'spec_discriminator'}
        disc_params = self._filter_top_keys(params_pure, disc_keys)
        gen_params = self._exclude_top_keys(params_pure, disc_keys)
        return gen_params, disc_params

    def _split_grads(self, grads_state: nnx.State) -> Tuple[Dict, Dict]:
        grads_pure = self._to_pure(grads_state)
        disc_keys = {'discriminator', 'spec_discriminator'}
        disc_grads = self._filter_top_keys(grads_pure, disc_keys)
        gen_grads = self._exclude_top_keys(grads_pure, disc_keys)
        return gen_grads, disc_grads

    def _ensure_optimizers(self):
        if self.gen_tx is not None:
            return
        # Schedules: keep constant for now, can be replaced with warmup/decay
        gen_lr = getattr(self.cfg.train, 'gen_optim_params', {}).get('lr', 1e-4) if hasattr(self.cfg, 'train') else 1e-4
        disc_lr = getattr(self.cfg.train, 'disc_optim_params', {}).get('lr', 1e-4) if hasattr(self.cfg, 'train') else 1e-4
        gen_grad_clip = getattr(self.cfg.train, 'gen_grad_clip', 1.0) if hasattr(self.cfg, 'train') else 1.0
        disc_grad_clip = getattr(self.cfg.train, 'disc_grad_clip', 1.0) if hasattr(self.cfg, 'train') else 1.0
        weight_decay = getattr(self.cfg.train, 'weight_decay', 0.0) if hasattr(self.cfg, 'train') else 0.0

        self.gen_tx = optax.chain(
            optax.clip_by_global_norm(gen_grad_clip),
            optax.adamw(gen_lr, b1=0.9, b2=0.999, weight_decay=weight_decay),
        )
        self.disc_tx = optax.chain(
            optax.clip_by_global_norm(disc_grad_clip),
            optax.adamw(disc_lr, b1=0.9, b2=0.999, weight_decay=weight_decay),
        )

        gen_params, disc_params = self._split_params()
        gen_param_arrays = self._state_arrays(gen_params)
        disc_param_arrays = self._state_arrays(disc_params)
        self.gen_opt_state = self.gen_tx.init(gen_param_arrays)
        self.disc_opt_state = self.disc_tx.init(disc_param_arrays)

    def _compute_losses(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Runs a forward pass and computes both generator and discriminator losses.

        This skeleton does not apply gradients yet; it validates the full
        forward+loss plumbing and returns a merged dict of loss metrics.
        """
        outputs = self.model(batch)
        disc_metrics = self.model.compute_disc_loss(outputs)
        gen_metrics = self.model.compute_gen_loss(outputs)

        # Merge metrics for convenience
        metrics = {**disc_metrics, **gen_metrics}
        return metrics

    def train_step(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        self._ensure_optimizers()

        # 1) Discriminator step
        def disc_loss_fn(model: CodecModule, batch_in: Dict[str, jnp.ndarray]):
            outputs = model(batch_in)
            disc_metrics = model.compute_disc_loss(outputs)
            return disc_metrics['disc_loss'], disc_metrics

        (disc_loss, disc_metrics), disc_grads = nnx.value_and_grad(disc_loss_fn, has_aux=True)(self.model, batch)

        # Split grads/params to discriminator subtree
        gen_params, disc_params = self._split_params()
        gen_grads, disc_grads_sub = self._split_grads(disc_grads)
        disc_param_arrays = self._state_arrays(disc_params)
        disc_grads_arrays = disc_grads_sub  # already arrays-like structure from grad

        disc_updates, self.disc_opt_state = self.disc_tx.update(disc_grads_arrays, self.disc_opt_state, disc_param_arrays)
        new_disc_params = optax.apply_updates(disc_param_arrays, disc_updates)
        # Apply discriminator updates to model
        nnx.update(self.model, nnx.State({'discriminator': new_disc_params.get('discriminator', {}),
                                          'spec_discriminator': new_disc_params.get('spec_discriminator', {})}))

        # 2) Generator step
        def gen_loss_fn(model: CodecModule, batch_in: Dict[str, jnp.ndarray]):
            outputs = model(batch_in)
            gen_metrics = model.compute_gen_loss(outputs)
            return gen_metrics['gen_loss'], gen_metrics

        (gen_loss, gen_metrics), gen_grads_all = nnx.value_and_grad(gen_loss_fn, has_aux=True)(self.model, batch)
        gen_grads_sub, _ = self._split_grads(gen_grads_all)
        gen_param_arrays = self._state_arrays(gen_params)
        gen_grads_arrays = gen_grads_sub

        gen_updates, self.gen_opt_state = self.gen_tx.update(gen_grads_arrays, self.gen_opt_state, gen_param_arrays)
        new_gen_params = optax.apply_updates(gen_param_arrays, gen_updates)
        # Apply generator updates: everything except discriminators
        nnx.update(self.model, nnx.State(new_gen_params))

        self.global_step += 1
        # Merge metric dicts for logging
        metrics = {**disc_metrics, **gen_metrics}
        metrics['disc_loss'] = disc_loss
        metrics['gen_loss'] = gen_loss
        return metrics

    def fit(self, data_iter: Iterable[Dict[str, jnp.ndarray]], steps: int = 1) -> None:
        """Unified DDP-style sharded training; works for 1 or many devices."""
        self._ensure_optimizers()
        self._setup_mesh()
        self._compile_sharded()

        graphdef, state = nnx.split(self.model)
        start_time = time.time()

        with self._mesh:
            for step_idx, batch in enumerate(data_iter):
                # Require global batch divisible by num devices
                B = batch['wav'].shape[0]
                ndev = self._mesh.devices.size
                if B % ndev != 0:
                    raise ValueError(f"Global batch {B} must be divisible by num devices {ndev}")

                # Place batch with sharding
                batch = {"wav": jax.device_put(batch['wav'], self._batch_sharding)}

                metrics, gen_grads, disc_grads = self._compute_grads_sharded(graphdef, state, batch)
                metrics_host = jax.device_get(metrics)
                gen_grads_host = jax.device_get(gen_grads)
                disc_grads_host = jax.device_get(disc_grads)

                # Split grads and params
                gen_grads_sub, _ = self._split_grads(gen_grads_host)
                _, disc_grads_only = self._split_grads(disc_grads_host)

                gen_params, disc_params = self._split_params()

                gen_updates, self.gen_opt_state = self.gen_tx.update(gen_grads_sub, self.gen_opt_state, gen_params)
                new_gen_params = optax.apply_updates(gen_params, gen_updates)

                disc_updates, self.disc_opt_state = self.disc_tx.update(disc_grads_only, self.disc_opt_state, disc_params)
                new_disc_params = optax.apply_updates(disc_params, disc_updates)

                # Merge and update model
                new_params = {**new_gen_params, **new_disc_params}
                nnx.update(self.model, nnx.State(new_params))

                self.global_step += 1
                # Log
                loss_str = ", ".join([f"{k}={float(metrics_host[k]):.4f}" for k in metrics_host.keys()])
                print(f"Step {self.global_step}: {loss_str}")

                if self._wandb_enabled:
                    wandb.log({k: float(v) for k, v in metrics_host.items()}, step=self.global_step)

                if step_idx + 1 >= steps:
                    break
        elapsed = time.time() - start_time
        print(f"Completed {steps} step(s) in {elapsed:.2f}s")

    # -------------------- Multi-device (sharded) training --------------------
    def _setup_mesh(self):
        if self._mesh is not None:
            return
        devices = mesh_utils.create_device_mesh((len(jax.devices()),))
        self._mesh = Mesh(devices, axis_names=('data',))
        self._batch_sharding = NamedSharding(self._mesh, P('data',))

    def _compile_sharded(self):
        if self._compute_grads_sharded is not None:
            return

        @partial(jax.jit,
                 in_shardings=(None, None, self._batch_sharding),
                 out_shardings=(None, None),
                 donate_argnums=(1,))
        def compute_grads_sharded(graphdef: nnx.GraphDef, state: nnx.State, batch: Dict[str, jnp.ndarray]):
            model = nnx.merge(graphdef, state)

            def disc_loss_fn(m: CodecModule, b):
                out = m(b)
                d = m.compute_disc_loss(out)
                return d['disc_loss'], d

            def gen_loss_fn(m: CodecModule, b):
                out = m(b)
                g = m.compute_gen_loss(out)
                return g['gen_loss'], g

            (disc_loss, disc_metrics), disc_grads = nnx.value_and_grad(disc_loss_fn, has_aux=True)(model, batch)
            (gen_loss, gen_metrics), gen_grads = nnx.value_and_grad(gen_loss_fn, has_aux=True)(model, batch)

            # Data-parallel average
            disc_grads = jax.tree.map(lambda x: jax.lax.pmean(x, 'data'), disc_grads)
            gen_grads = jax.tree.map(lambda x: jax.lax.pmean(x, 'data'), gen_grads)

            metrics = {
                'disc_loss': disc_loss,
                'gen_loss': gen_loss,
            }
            return metrics, gen_grads, disc_grads

        self._compute_grads_sharded = compute_grads_sharded

    # fit_sharded removed; fit is the unified path


