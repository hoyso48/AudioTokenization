import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import sys
sys.path.insert(0, '/home/hoyeol')

from trainer_nnx import CodecTrainer


def _dummy_data_iterator(batch_size: int, time_steps: int, steps: int):
    for _ in range(steps):
        wav = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
        yield {"wav": wav}


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Force CPU for safest smoke-run
    # jax.config.update("jax_platform_name", "cpu")
    import time
    from tqdm import tqdm

    # Device diagnostics
    try:
        print(f"JAX backend={jax.default_backend()} device_count={jax.device_count()} local_device_count={jax.local_device_count()}")
        print(f"Devices: {jax.devices()}")
    except Exception:
        pass

    trainer = CodecTrainer(cfg=cfg, rng_seed=0)

    # Very small shapes for quick, light test; ensure batch divisible by device count
    # Keep smoke small and configurable
    steps = int(getattr(cfg.train.trainer, 'smoke_steps', 200))
    ndev = max(1, jax.local_device_count())
    # Larger per-device batch improves TPU utilization
    per_device = 16
    batch_size = max(per_device * ndev, ndev)
    if batch_size % ndev != 0:
        batch_size = ((batch_size + ndev - 1) // ndev) * ndev
    print(f"Using batch_size={batch_size} across {ndev} local devices")
    data_iter = _dummy_data_iterator(batch_size=batch_size, time_steps=64000, steps=steps)

    start = time.time()
    trainer.fit(data_iter=data_iter, steps=steps)
    elapsed = time.time() - start
    if jax.process_index() == 0 and elapsed > 0:
        it_per_s = steps / elapsed
        print(f"Throughput: {it_per_s:.2f} it/s over {steps} steps")


if __name__ == "__main__":
    # import multiprocessing as mp
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass
    main()


