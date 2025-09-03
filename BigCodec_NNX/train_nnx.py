import hydra
from omegaconf import DictConfig

from trainer_nnx import CodecTrainer
from data_module_nnx import DataModuleNNX


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Minimal run with real Grain-based data loader
    trainer = CodecTrainer(cfg=cfg, rng_seed=0)
    dm = DataModuleNNX(cfg)
    train_loader = dm.train_dataloader()
    # Determine steps: prefer max_steps, else epochs * steps_per_epoch
    try:
        steps_per_epoch = int(dm.steps_per_epoch('train'))
    except Exception:
        steps_per_epoch = 0
    max_steps = getattr(cfg.train.trainer, 'max_steps', None)
    if isinstance(max_steps, int) and max_steps > 0:
        steps = max_steps
    else:
        max_epochs = int(getattr(cfg.train.trainer, 'max_epochs', 1))
        steps = max(steps_per_epoch * max_epochs, 1) if steps_per_epoch > 0 else 1
    # Unified fit uses sharded path under the hood (1+ devices)
    trainer.fit(data_iter=train_loader, steps=steps)


if __name__ == "__main__":
    main()


