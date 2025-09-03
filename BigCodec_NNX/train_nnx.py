import hydra
from omegaconf import DictConfig

from AudioTokenization.BigCodec_NNX.trainer_nnx import CodecTrainer
from AudioTokenization.BigCodec_NNX.data_module_nnx import DataModuleNNX


@hydra.main(config_path="../CP/config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Minimal run with real Grain-based data loader
    trainer = CodecTrainer(cfg=cfg, rng_seed=0)
    dm = DataModuleNNX(cfg)
    train_loader = dm.train_dataloader()
    # Unified fit uses sharded path under the hood (1+ devices)
    trainer.fit(data_iter=train_loader, steps=2)


if __name__ == "__main__":
    main()


