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
from tqdm import tqdm
import os
import orbax.checkpoint as ocp

from codec_module import CodecModule
from metrics_nnx import (
    CodebookMetrics,
    PESQMetric,
    STOIMetric,
    SISNRJAXMetric,
    SISDRJAXMetric,
    MelJAXMetric,
    AvgSimJAXMetric,
    MeanMetric,
)
from data_module_nnx import DataModuleNNX

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

        # W&B (optional but fail visibly)
        self._wandb_enabled = True
        project = getattr(self.cfg, 'project', 'Audio-Tokenizer-NNX')
        name = getattr(self.cfg, 'name', 'nnx-run')
        save_dir = getattr(self.cfg, 'log_dir', None)
        run_id = getattr(self.cfg, 'wandb_id', None)
        if jax.process_index() == 0:
            try:
                wandb.init(project=project, name=name, dir=save_dir, id=run_id, resume='allow')
            except Exception as e:
                print(f"[Warning] W&B init failed: {e}. Proceeding with logging disabled.")
                self._wandb_enabled = False
        else:
            self._wandb_enabled = False

        # Sharded compute cache
        self._mesh: Optional[Mesh] = None
        self._batch_sharding: Optional[NamedSharding] = None
        self._compute_grads_sharded = None
        self._train_step_sharded = None
        self._val_step_compiled = None
        # Checkpointing
        self._ckpt_mgr = None
        self._best_val = None
        self._best_path = None

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

    def _collect_log_payload(self, metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Collect minimal scalar payload from device metrics and host LRs."""
        to_fetch = {
            'disc_loss': metrics['disc_loss'],
            'gen_loss': metrics['gen_loss'],
        }
        optional_keys = [
            'disc_real_loss', 'disc_fake_loss',
            'mel_loss', 'adv_loss', 'fm_loss', 'spec_fm_loss',
            'vq_loss', 'avg_sim',
        ]
        for k in optional_keys:
            if k in metrics:
                to_fetch[k] = metrics[k]
        metrics_host = jax.device_get(to_fetch)
        payload = {k: float(v) for k, v in metrics_host.items()}
        if hasattr(self, '_gen_schedule_fn'):
            payload['gen_lr'] = float(self._gen_schedule_fn(self.global_step))
        if hasattr(self, '_disc_schedule_fn'):
            payload['disc_lr'] = float(self._disc_schedule_fn(self.global_step))
        return payload

    def _init_orbax_manager(self) -> None:
        base_dir = getattr(self.cfg, 'log_dir', './pl_log')
        ckpt_dir = os.path.abspath(os.path.join(base_dir, 'nnx_checkpoints'))
        # Legacy API requires a Checkpointer, not a raw handler
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self._ckpt_mgr = ocp.CheckpointManager(
            ckpt_dir,
            checkpointer,
            options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True),
        )

    def _maybe_restore_checkpoint(self, gen_params: nnx.State, disc_params: nnx.State) -> Tuple[nnx.State, nnx.State]:
        restore_id = getattr(self.cfg, 'resume_ckpt', None)
        if not restore_id:
            return gen_params, disc_params
        try:
            if isinstance(restore_id, str) and restore_id.lower() == 'latest':
                step = self._ckpt_mgr.latest_step()
            elif isinstance(restore_id, (int, float)) or (isinstance(restore_id, str) and restore_id.isdigit()):
                step = int(restore_id)
            else:
                step = None
            if step is not None:
                state = self._ckpt_mgr.restore(step)
                if isinstance(state, dict):
                    gen_params = state.get('gen_params', gen_params)
                    disc_params = state.get('disc_params', disc_params)
                    self.gen_opt_state = state.get('gen_opt_state', self.gen_opt_state)
                    self.disc_opt_state = state.get('disc_opt_state', self.disc_opt_state)
                    self.global_step = int(state.get('global_step', self.global_step))
                    nnx.update(self.model, nnx.merge_state(gen_params, disc_params))
        except Exception as e:
            print(f"[Warning] Failed to restore from checkpoint id {restore_id}: {e}")
        return gen_params, disc_params

    def _ensure_optimizers(self):
        if self.gen_tx is not None:
            return
        # Optimizer params
        gen_opt = getattr(self.cfg.train, 'gen_optim_params', {}) if hasattr(self.cfg, 'train') else {}
        disc_opt = getattr(self.cfg.train, 'disc_optim_params', {}) if hasattr(self.cfg, 'train') else {}
        gen_grad_clip = float(getattr(self.cfg.train, 'gen_grad_clip', 1.0)) if hasattr(self.cfg, 'train') else 1.0
        disc_grad_clip = float(getattr(self.cfg.train, 'disc_grad_clip', 1.0)) if hasattr(self.cfg, 'train') else 1.0
        weight_decay = float(getattr(self.cfg.train, 'weight_decay', 0.01)) if hasattr(self.cfg, 'train') else 0.0

        # Betas
        gen_betas = tuple(gen_opt.get('betas', (0.9, 0.999)))
        disc_betas = tuple(disc_opt.get('betas', (0.9, 0.999)))

        # Learning rate schedules (Warmup + Linear decay)
        def build_schedule(sched_cfg, default_max_lr: float):
            warmup = int(sched_cfg.get('warmup_step', 0))
            down = int(sched_cfg.get('down_step', 0))
            min_lr = float(sched_cfg.get('min_lr', default_max_lr))
            max_lr = float(sched_cfg.get('max_lr', default_max_lr))
            warmup_fn = optax.linear_schedule(init_value=0.0, end_value=max_lr, transition_steps=warmup)
            decay_fn = optax.linear_schedule(init_value=max_lr, end_value=min_lr, transition_steps=down)
            return optax.join_schedules([warmup_fn, decay_fn], boundaries=[warmup])

        gen_sched_cfg = getattr(self.cfg.train, 'gen_schedule_params', {}) if hasattr(self.cfg, 'train') else {}
        disc_sched_cfg = getattr(self.cfg.train, 'disc_schedule_params', {}) if hasattr(self.cfg, 'train') else {}
        gen_schedule = build_schedule(gen_sched_cfg, float(gen_opt.get('lr', 1e-4)))
        disc_schedule = build_schedule(disc_sched_cfg, float(disc_opt.get('lr', 1e-4)))

        self.gen_tx = optax.chain(
            optax.clip_by_global_norm(gen_grad_clip),
            optax.adamw(gen_schedule, b1=gen_betas[0], b2=gen_betas[1], weight_decay=weight_decay),
        )
        self.disc_tx = optax.chain(
            optax.clip_by_global_norm(disc_grad_clip),
            optax.adamw(disc_schedule, b1=disc_betas[0], b2=disc_betas[1], weight_decay=weight_decay),
        )

        # Keep schedule fns for host-side LR logging
        self._gen_schedule_fn = gen_schedule
        self._disc_schedule_fn = disc_schedule

        # Initialize optimizer states using nnx.State-shaped trees
        params_state = nnx.state(self.model, nnx.Param)
        params_pure = self._to_pure(params_state)
        disc_keys = {'discriminator', 'spec_discriminator'}
        gen_params_state = nnx.State(self._exclude_top_keys(params_pure, disc_keys))
        disc_params_state = nnx.State(self._filter_top_keys(params_pure, disc_keys))
        self.gen_opt_state = self.gen_tx.init(self._state_arrays(gen_params_state))
        self.disc_opt_state = self.disc_tx.init(self._state_arrays(disc_params_state))

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

        graphdef, _, other_states = nnx.split(self.model, nnx.Param, ...)
        params_state = nnx.state(self.model, nnx.Param)
        params_pure = self._to_pure(params_state)
        disc_keys = {'discriminator', 'spec_discriminator'}
        gen_params = nnx.State(self._exclude_top_keys(params_pure, disc_keys))
        disc_params = nnx.State(self._filter_top_keys(params_pure, disc_keys))

        # Discriminator grads only w.r.t discriminator params
        def disc_loss_fn(disc_p: nnx.State, gen_p: nnx.State, b: Dict[str, jnp.ndarray]):
            current_state = nnx.merge_state(gen_p, disc_p, other_states)
            m = nnx.merge(graphdef, current_state)
            out = m(b)
            d = m.compute_disc_loss(out)
            return d['disc_loss']

        disc_loss, disc_grads = jax.value_and_grad(disc_loss_fn)(disc_params, gen_params, batch)
        disc_updates, self.disc_opt_state = self.disc_tx.update(disc_grads, self.disc_opt_state, disc_params)
        disc_params = optax.apply_updates(disc_params, disc_updates)

        # Generator grads only w.r.t generator params
        def gen_loss_fn(gen_p: nnx.State, disc_p: nnx.State, b: Dict[str, jnp.ndarray]):
            current_state = nnx.merge_state(gen_p, disc_p, other_states)
            m = nnx.merge(graphdef, current_state)
            out = m(b)
            g = m.compute_gen_loss(out)
            return g['gen_loss']

        gen_loss, gen_grads = jax.value_and_grad(gen_loss_fn)(gen_params, disc_params, batch)
        gen_updates, self.gen_opt_state = self.gen_tx.update(gen_grads, self.gen_opt_state, gen_params)
        gen_params = optax.apply_updates(gen_params, gen_updates)

        # Apply updated params to live model
        nnx.update(self.model, nnx.merge_state(gen_params, disc_params))

        self.global_step += 1
        return {
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
        }

    def fit(self, data_iter: Iterable[Dict[str, jnp.ndarray]], steps: int = 1) -> None:
        """Unified DDP-style sharded training; works for 1 or many devices."""
        self._ensure_optimizers()
        self._setup_mesh()
        self._compile_train_step()
        # Validation cadence controls
        check_every = 0
        try:
            check_every = int(getattr(self.cfg.train.trainer, 'check_val_every_n_epoch', 0))
        except Exception:
            check_every = 0
        try:
            steps_per_epoch = int(DataModuleNNX(self.cfg).steps_per_epoch('train'))
        except Exception:
            steps_per_epoch = 0
        # Initialize param subtrees once from current model
        full_params_state = nnx.state(self.model, nnx.Param)
        params_pure = self._to_pure(full_params_state)
        disc_keys = {'discriminator', 'spec_discriminator'}
        gen_params = nnx.State(self._exclude_top_keys(params_pure, disc_keys))
        disc_params = nnx.State(self._filter_top_keys(params_pure, disc_keys))
        # Initialize optimizer states bound to the corresponding param subtrees
        if self.gen_tx is None or self.disc_tx is None:
            self._ensure_optimizers()
        if self.gen_opt_state is None:
            self.gen_opt_state = self.gen_tx.init(self._state_arrays(gen_params))
        if self.disc_opt_state is None:
            self.disc_opt_state = self.disc_tx.init(self._state_arrays(disc_params))
        start_time = time.time()
        last_tic = start_time
        progress = None
        if jax.process_index() == 0:
            progress = tqdm(total=steps, desc="Training", dynamic_ncols=True, leave=True)

        with self._mesh:
            # Setup/resume checkpoint manager
            if self._ckpt_mgr is None:
                self._init_orbax_manager()
                gen_params, disc_params = self._maybe_restore_checkpoint(gen_params, disc_params)
            for step_idx, batch in enumerate(data_iter):
                # Require global batch divisible by num devices
                B = batch['wav'].shape[0]
                ndev = self._mesh.devices.size
                if B % ndev != 0:
                    raise ValueError(f"Global batch {B} must be divisible by num devices {ndev}")

                # Place batch with sharding (keep transfer minimal)
                batch = {"wav": jax.device_put(batch['wav'], self._batch_sharding)}

                # On-device single-step update with param-subtree grads only
                gen_params, disc_params, self.gen_opt_state, self.disc_opt_state, metrics = self._train_step_sharded(
                    gen_params, disc_params, self.gen_opt_state, self.disc_opt_state, batch
                )

                # Apply updated params to live model periodically (cheap on host)
                nnx.update(self.model, nnx.merge_state(gen_params, disc_params))

                self.global_step += 1

                # Sparse logging inside training loop for readability
                cfg_log_every = int(getattr(self.cfg.train.trainer, 'log_every_n_steps', 100))
                if self.global_step % cfg_log_every == 0 and jax.process_index() == 0:
                    payload = self._collect_log_payload(metrics)
                    if progress is not None:
                        pf = {
                            'disc_loss': f"{payload.get('disc_loss', 0.0):.4f}",
                            'gen_loss': f"{payload.get('gen_loss', 0.0):.4f}",
                        }
                        if 'gen_lr' in payload and 'disc_lr' in payload:
                            pf.update({'gen_lr': f"{payload['gen_lr']:.2E}", 'disc_lr': f"{payload['disc_lr']:.2E}"})
                        progress.set_postfix(pf)
                    if self._wandb_enabled:
                        wandb.log(payload, step=self.global_step)

                if step_idx + 1 >= steps:
                    break
                if progress is not None:
                    progress.update(1)

                # Validation cadence by epoch
                if check_every and steps_per_epoch > 0 and (self.global_step % (steps_per_epoch * check_every) == 0):
                    self._run_validation()
                    # Save 'last' checkpoint after validation
                    try:
                        payload = {
                            'gen_params': gen_params,
                            'disc_params': disc_params,
                            'gen_opt_state': self.gen_opt_state,
                            'disc_opt_state': self.disc_opt_state,
                            'global_step': self.global_step,
                        }
                        self._ckpt_mgr.save(self.global_step, payload)
                    except Exception as e:
                        print(f"[Warning] Failed to save checkpoint at step {self.global_step}: {e}")
        elapsed = time.time() - start_time
        if progress is not None:
            progress.close()
        print(f"Completed {steps} step(s) in {elapsed:.2f}s")
        # Run final test evaluation
        try:
            self._run_test()
        except Exception as e:
            print(f"[Warning] Test phase failed: {e}")

    # -------------------- Multi-device (sharded) training --------------------
    def _setup_mesh(self):
        if self._mesh is not None:
            return
        devices = mesh_utils.create_device_mesh((len(jax.devices()),))
        self._mesh = Mesh(devices, axis_names=('data',))
        self._batch_sharding = NamedSharding(self._mesh, P('data',))

    def _compile_sharded(self):
        # Deprecated: replaced by on-device single-step updates
        return

    def _compile_train_step(self):
        if self._train_step_sharded is not None:
            return

        # Cache graphdef and non-param states for reconstruction inside jit
        graphdef, params_init, other_states = nnx.split(self.model, nnx.Param, ...)

        @partial(
            jax.jit,
            in_shardings=(None, None, None, None, self._batch_sharding),
            out_shardings=(None, None, None, None, None),
            donate_argnums=(0, 1, 2, 3),
        )
        def train_step(gen_params: nnx.State,
                       disc_params: nnx.State,
                       gen_opt_state: optax.OptState,
                       disc_opt_state: optax.OptState,
                       batch: Dict[str, jnp.ndarray]):
            # Discriminator grads only w.r.t discriminator params
            def disc_loss_fn(disc_p: nnx.State, gen_p: nnx.State, b: Dict[str, jnp.ndarray]):
                current_state = nnx.merge_state(gen_p, disc_p, other_states)
                m = nnx.merge(graphdef, current_state)
                out = m(b)
                d = m.compute_disc_loss(out)
                aux = {
                    'disc_real_loss': d['real_loss'],
                    'disc_fake_loss': d['fake_loss'],
                }
                return d['disc_loss'], aux

            (disc_loss, disc_aux), disc_grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(disc_params, gen_params, batch)
            disc_updates, disc_opt_state = self.disc_tx.update(disc_grads, disc_opt_state, disc_params)
            disc_params = optax.apply_updates(disc_params, disc_updates)

            # Generator grads only w.r.t generator params
            def gen_loss_fn(gen_p: nnx.State, disc_p: nnx.State, b: Dict[str, jnp.ndarray]):
                current_state = nnx.merge_state(gen_p, disc_p, other_states)
                m = nnx.merge(graphdef, current_state)
                out = m(b)
                g = m.compute_gen_loss(out)
                aux = {k: v for k, v in g.items() if k != 'gen_loss'}
                if 'avg_sim' in out:
                    aux['avg_sim'] = jnp.mean(out['avg_sim'])
                return g['gen_loss'], aux

            (gen_loss, gen_aux), gen_grads = jax.value_and_grad(gen_loss_fn, has_aux=True)(gen_params, disc_params, batch)
            gen_updates, gen_opt_state = self.gen_tx.update(gen_grads, gen_opt_state, gen_params)
            gen_params = optax.apply_updates(gen_params, gen_updates)

            metrics = {
                'disc_loss': disc_loss,
                'gen_loss': gen_loss,
            }
            # Merge aux metrics without increasing host transfers beyond one dict
            for k, v in disc_aux.items():
                metrics[k] = v
            for k, v in gen_aux.items():
                metrics[k] = v
            return gen_params, disc_params, gen_opt_state, disc_opt_state, metrics

        self._train_step_sharded = train_step

    # -------------------- Validation --------------------
    def _compile_val_step(self):
        if self._val_step_compiled is not None:
            return
        graphdef, _, other_states = nnx.split(self.model, nnx.Param, ...)

        @partial(
            jax.jit,
            in_shardings=(self._batch_sharding,),
            out_shardings=(None,),
        )
        def val_step(batch: Dict[str, jnp.ndarray]):
            m = nnx.merge(graphdef, other_states)
            out = m(batch)
            return out

        self._val_step_compiled = val_step

    def _run_validation(self):
        dm = DataModuleNNX(self.cfg)
        val_loader = dm.val_dataloader()
        sample_rate = int(self.cfg.dataset.sample_rate) if hasattr(self.cfg, 'dataset') else 16000

        self._compile_val_step()

        # Aggregators
        cb = CodebookMetrics(codebook_size=int(self.cfg.model.codec_decoder.codebook_size))
        pesq_m = PESQMetric()
        stoi_m = STOIMetric()

        n_batches = 0
        si_snr_m = SISNRJAXMetric()
        si_sdr_m = SISDRJAXMetric()
        mel_m = MelJAXMetric()
        avg_sim_m = AvgSimJAXMetric()

        for batch in val_loader:
            batch = {"wav": jax.device_put(batch['wav'], self._batch_sharding)}
            out = self._val_step_compiled(batch)
            y = out['gt_wav']
            y_ = out['gen_wav']
            si_snr_m.update_pair(y_, y)
            si_sdr_m.update_pair(y_, y)
            mel_m.update_pair(y_, y)
            avg_sim_m.update_tensor(out.get('avg_sim', None))
            vq_code = out.get('vq_code', None)
            cb.update(vq_code)
            # CPU metrics per batch (optional if libs are installed)
            try:
                pesq_m.update_pair(jax.device_get(y_), jax.device_get(y), sample_rate)
            except Exception:
                pass
            try:
                stoi_m.update_pair(jax.device_get(y_), jax.device_get(y), sample_rate)
            except Exception:
                pass

            n_batches += 1

        if n_batches == 0:
            return

        metrics = {
            'val/si_snr': si_snr_m.compute(),
            'val/si_sdr': si_sdr_m.compute(),
            'val/mel_loss': mel_m.compute(),
        }
        avg_sim_val = avg_sim_m.compute()
        if avg_sim_val != 0.0:
            metrics['val/avg_sim'] = avg_sim_val
        metrics.update({f"val/{k}": v for k, v in cb.compute().items()})
        pesq_val = pesq_m.compute()
        stoi_val = stoi_m.compute()
        if pesq_val > 0:
            metrics['val/pesq'] = pesq_val
        if stoi_val > 0:
            metrics['val/stoi'] = stoi_val

        if self._wandb_enabled:
            wandb.log(metrics, step=self.global_step)

    # fit_sharded removed; fit is the unified path

    def _run_test(self):
        dm = DataModuleNNX(self.cfg)
        test_loader = dm.test_dataloader()
        sample_rate = int(self.cfg.dataset.sample_rate) if hasattr(self.cfg, 'dataset') else 16000

        self._compile_val_step()

        cb = CodebookMetrics(codebook_size=int(self.cfg.model.codec_decoder.codebook_size))
        pesq_m = PESQMetric()
        stoi_m = STOIMetric()
        si_snr_m = SISNRJAXMetric()
        si_sdr_m = SISDRJAXMetric()
        mel_m = MelJAXMetric()
        avg_sim_m = AvgSimJAXMetric()

        n_batches = 0
        for batch in test_loader:
            batch = {"wav": jax.device_put(batch['wav'], self._batch_sharding)}
            out = self._val_step_compiled(batch)
            y = out['gt_wav']
            y_ = out['gen_wav']
            si_snr_m.update_pair(y_, y)
            si_sdr_m.update_pair(y_, y)
            mel_m.update_pair(y_, y)
            avg_sim_m.update_tensor(out.get('avg_sim', None))
            cb.update(out.get('vq_code', None))
            try:
                pesq_m.update_pair(jax.device_get(y_), jax.device_get(y), sample_rate)
            except Exception:
                pass
            try:
                stoi_m.update_pair(jax.device_get(y_), jax.device_get(y), sample_rate)
            except Exception:
                pass
            n_batches += 1

        if n_batches == 0:
            return

        metrics = {
            'test/si_snr': si_snr_m.compute(),
            'test/si_sdr': si_sdr_m.compute(),
            'test/mel_loss': mel_m.compute(),
        }
        avg_sim_val = avg_sim_m.compute()
        if avg_sim_val != 0.0:
            metrics['test/avg_sim'] = avg_sim_val
        metrics.update({f"test/{k}": v for k, v in cb.compute().items()})
        pesq_val = pesq_m.compute()
        stoi_val = stoi_m.compute()
        if pesq_val > 0:
            metrics['test/pesq'] = pesq_val
        if stoi_val > 0:
            metrics['test/stoi'] = stoi_val

        if self._wandb_enabled:
            wandb.log(metrics, step=self.global_step)

