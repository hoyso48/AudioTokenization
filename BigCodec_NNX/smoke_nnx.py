import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import sys
sys.path.insert(0, '/home/hoyeol')

from AudioTokenization.BigCodec_NNX.trainer_nnx import CodecTrainer


def _dummy_data_iterator(batch_size: int, time_steps: int, steps: int):
    for _ in range(steps):
        wav = jnp.zeros((batch_size, time_steps), dtype=jnp.float32)
        yield {"wav": wav}


@hydra.main(config_path="../CP/config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Force CPU for safest smoke-run
    # jax.config.update("jax_platform_name", "cpu")

    trainer = CodecTrainer(cfg=cfg, rng_seed=0)

    # Very small shapes for quick, light test
    data_iter = _dummy_data_iterator(batch_size=4, time_steps=16000, steps=2)
    trainer.fit(data_iter=data_iter, steps=2)


if __name__ == "__main__":
    main()


