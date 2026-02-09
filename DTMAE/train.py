import os
import pytorch_lightning as pl
import hydra
import torch
import random
import time
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from data_module import DataModule
from lightning_module import CodecLightningModule

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='default', version_base=None)
def train(cfg):
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.log_dir, 
                            save_top_k=1, save_last=True,
                            every_n_train_steps=10000, monitor='mel_loss', mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    datamodule = DataModule(cfg)
    lightning_module = CodecLightningModule(cfg)

    devices = cfg.train.trainer.get("devices", 1)
    if isinstance(devices, (list, tuple)):
        use_ddp = len(devices) >= 2
    elif isinstance(devices, int):
        use_ddp = devices >= 2
    else:
        use_ddp = False
    
    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True) if use_ddp else None,
        callbacks=callbacks,
        limit_train_batches=1.0 if not cfg.debug else 0.001,
        logger = pl.loggers.WandbLogger(project='Audio-Tokenizer', name=cfg.name, save_dir=cfg.log_dir, id=cfg.wandb_id),
    )
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.resume_ckpt)
    if cfg.ckpt is not None:
        lightning_module = CodecLightningModule.load_from_checkpoint(cfg.ckpt)
        lightning_module.eval()
    # print(lightning_module.dtp.log_tau)
    # print(torch.tensor(torch.log(torch.tensor(0.09))))
    # lightning_module.dtp.log_tau.data = torch.tensor(torch.log(torch.tensor(0.09)))
    trainer.validate(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)
    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')
    
if __name__ == '__main__':
    train()
