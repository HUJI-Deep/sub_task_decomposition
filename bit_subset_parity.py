import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2LMHeadModel, GPT2Config


class BitSubsetParity(pl.LightningModule):
    def __init__(self, step_by_step: bool, num_of_bits: int, width=512, num_heads=8, depth=3, learning_rate=1e-3, warmup_steps=1000, weight_decay=1e-2, evaluate_with_greedy_decoding=False):
        super().__init__()
        self.save_hyperparameters()
        self.step_by_step = step_by_step
        self.num_of_bits = num_of_bits
        self.evaluate_with_greedy_decoding = evaluate_with_greedy_decoding
        self.generation_length = ((self.num_of_bits * 3) // 2 - 2) if self.step_by_step else self.num_of_bits
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=self.generation_length, n_embd=width, n_layer=depth, n_head=num_heads, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=2, eos_token_id=2))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        inputs=inputs.long()
        if self.step_by_step:
            do_sample = not self.evaluate_with_greedy_decoding
            inputs = self.model.generate(inputs, do_sample=do_sample, max_length=self.generation_length, min_length=self.generation_length, pad_token_id=2, num_beams=1)
        logits = self.model(inputs).logits[:, self.num_of_bits - 1:, :2]
        predictions = torch.argmax(logits, dim=2)[:, 0]
        return predictions

    def _training_evaluation_common(self, batch):
        batch['label'] = batch['label'].long()
        logits = self.model(batch['input_ids'].long()).logits[:, -batch['label'].shape[1]:, :2]
        loss = self.loss(logits.permute(0, 2, 1), batch['label'])
        predictions = torch.argmax(logits, dim=2)
        accuracy_with_steps = torch.mean((predictions == batch['label']).float())
        final_label_accuracy =torch.mean((predictions[:, -1] == batch['label'][:, -1]).float())
        return loss, final_label_accuracy, accuracy_with_steps

    def training_step(self, batch, batch_idx):
        loss, final_label_accuracy, accuracy_with_steps = self._training_evaluation_common(batch)
        self.log("loss/train", loss)
        self.log("accuracy/train", final_label_accuracy)
        if self.step_by_step:
            self.log("accuracy_with_steps/train", accuracy_with_steps)
        return loss

    def _prepare_batch_for_evaluation(self, batch):
        if self.step_by_step:
            do_sample = not self.evaluate_with_greedy_decoding
            batch['input_ids'] = self.model.generate(batch['input_ids'].long(), do_sample=do_sample, max_length=self.generation_length, min_length=self.generation_length, pad_token_id=2).detach()
        return batch

    def validation_step(self, batch, batch_idx):
        loss, accuracy, _ = self._training_evaluation_common(self._prepare_batch_for_evaluation(batch))
        self.log("val_loss", loss)
        self.log("loss/val", loss)
        self.log("accuracy/val", accuracy)

    def test_step(self, batch, batch_idx):
        loss, accuracy, _ = self._training_evaluation_common(self._prepare_batch_for_evaluation(batch))
        self.log("loss/test", loss)
        self.log("accuracy/test", accuracy)

    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        start_factor=1e-2
        lr_lambda = lambda epoch: (start_factor +
                (1. - start_factor) * min(self.hparams.warmup_steps, epoch) / self.hparams.warmup_steps)
        lr_scheduler = LambdaLR(optimizer, lr_lambda)        
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--width', type=int, default=512)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--depth', type=int, default=3)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--evaluate_with_greedy_decoding', dest='evaluate_with_greedy_decoding', action='store_true')
        parser.add_argument('--evaluate_with_sampling', dest='evaluate_with_greedy_decoding', action='store_false')
        parser.set_defaults(evaluate_with_greedy_decoding=False)
        return parser
