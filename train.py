import argparse
import os
import wandb
from pytorch_lightning.loggers import WandbLogger as Logger
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from sconf import Config
import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin


def train(config: Config):
    pl.seed_everything(config.seed_everything, workers=True)
    # print(dict(config.model))
    model_module = LitCoMER(
        **dict(config.model)
    )
    data_module = CROHMEDatamodule(
        **dict(config.data),
    )
    # logger = Logger(name=config.wandb.name,
    #                 project=config.wandb.project,
    #                 log_model=config.wandb.log_model,
    #                 config=dict(config),
    #                 )
    # logger.watch(model_module,
    #              log="all",
    #              log_freq=100
    #              )

    lr_callback = pl.callbacks.LearningRateMonitor(
        **dict(config.trainer.callbacks[0].init_args),
        # logging_interval=config.trainer.callbacks[0].init_args.logging_interval
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **dict(config.trainer.callbacks[1].init_args),
        # save_top_k=config.trainer.callbacks[1].init_args.save_top_k,
        # monitor=config.trainer.callbacks[1].init_args.monitor,
        # mode=config.trainer.callbacks[1].init_args.mode,
        # filename=config.trainer.callbacks[1].init_args.filename
    )

    trainer = pl.Trainer(
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        limit_train_batches=1.0,

        gpus=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        deterministic=config.trainer.deterministic,

        plugins=DDPPlugin(find_unused_parameters=False),
        # logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    train(config)
