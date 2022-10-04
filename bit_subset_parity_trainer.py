import argparse
import json
import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from bit_subset_parity import BitSubsetParity
from bit_subset_parity_data_module import BitSubsetParityDataModule


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BitSubsetParity.add_model_specific_args(parser)
    parser = BitSubsetParityDataModule.add_data_module_specific_args(parser)
    return parser.parse_args() 


def main():
    pl.seed_everything(1234)
    args = parse_arguments()
    data_module = BitSubsetParityDataModule(step_by_step=args.step_by_step,
                                            max_training_steps=args.max_steps * args.accumulate_grad_batches,
                                            num_of_bits=args.num_of_bits,
                                            train_batch_size=args.train_batch_size,
                                            eval_batch_size=args.eval_batch_size,
                                            eval_steps=args.eval_steps,
                                            num_workers=args.num_workers,
                                            seed=args.seed)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    model = BitSubsetParity(step_by_step=args.step_by_step,
                            num_of_bits=args.num_of_bits,
                            width=args.width,
                            num_heads=args.num_heads,
                            depth=args.depth,
                            learning_rate=args.learning_rate,
                            warmup_steps=args.warmup_steps,
                            weight_decay=args.weight_decay,
                            evaluate_with_greedy_decoding=args.evaluate_with_greedy_decoding)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, data_module)
    print("Training finished")
    last_model_test_results = trainer.test(model, datamodule=data_module, ckpt_path=None)
    best_model_test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")
    with open(os.path.join(trainer.log_dir, "last_model_test_results.json"), 'w') as f:
        json.dump(last_model_test_results, f)
    with open(os.path.join(trainer.log_dir, "best_model_test_results.json"), 'w') as f:
        json.dump(best_model_test_results, f)
    shutil.rmtree(checkpoint_callback.dirpath)


if __name__ == '__main__':
    main()
