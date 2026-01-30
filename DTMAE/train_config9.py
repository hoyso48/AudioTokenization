import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data_module import DataModule
from lightning_module import CodecLightningModule

seed_everything(1024)


@hydra.main(config_path="config9", config_name="default", version_base=None)
def train(cfg):
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log_dir,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=10000,
        monitor="mel_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    datamodule = DataModule(cfg)
    lightning_module = CodecLightningModule(cfg)

    trainer = pl.Trainer(
        **cfg.train.trainer,
        callbacks=callbacks,
        limit_train_batches=1.0 if not cfg.debug else 0.001,
        logger=pl.loggers.WandbLogger(
            project="Audio-Tokenizer", name=cfg.name, save_dir=cfg.log_dir, id=cfg.wandb_id
        ),
    )
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.resume_ckpt)
    trainer.validate(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)
    print(
        f"Training ends, best score: {checkpoint_callback.best_model_score}, "
        f"ckpt path: {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    train()


