import argparse
import itertools

import numpy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class BitSubsetParityDataset(Dataset):
    def __init__(self, x, subset, step_by_step: bool = False):
        self.x = x
        self.subset = numpy.copy(subset)
        self.step_by_step = step_by_step
        if self.step_by_step:
            self.num_bits = self.x.shape[1]
            binary_tree_bfs = self.num_bits + numpy.arange(2 * (self.num_bits // 4 - 1))
            self.subset = numpy.concatenate((self.subset, binary_tree_bfs))
            self.subset = self.subset.reshape((-1, 2))

    def __getitem__(self, index):        
        input_ids = self.x[index]
        if self.step_by_step:            
            input_ids = numpy.concatenate((input_ids, numpy.zeros((len(input_ids) // 2 - 1,), dtype=bool)))
            for i, (x, y) in enumerate(self.subset):                
                input_ids[self.num_bits + i] = numpy.logical_xor(input_ids[x], input_ids[y])
            label = input_ids[self.num_bits:]
            input_ids = input_ids[:-1]
        else:
            label = numpy.expand_dims((input_ids[self.subset].astype(numpy.long).sum() % 2).astype(bool), axis=0)
        return {"input_ids": input_ids, "label": label}

    def __len__(self):
        return len(self.x)


class BitSubsetParityDataModule(LightningDataModule):
    def __init__(
        self,
        step_by_step: bool = False,
        num_of_bits: int = 64,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        max_training_steps: int = 100000,
        eval_steps: int = 32,
        num_workers=1,
        seed=42,
        **kwargs,
    ):
        super().__init__()
        self.max_training_steps = max_training_steps
        self.eval_steps = eval_steps
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.step_by_step = step_by_step
        self.num_of_bits = num_of_bits
        self.seed = seed

    def _build_dataset_held_out_at_input_level(self):
        random_state = RandomState(MT19937(SeedSequence(self.seed)))
        self.subset = random_state.permutation(self.num_of_bits)[:self.num_of_bits // 2]
        datasets_size = self.train_batch_size*self.max_training_steps + 2*self.eval_batch_size*self.eval_steps
        if datasets_size < 2 ** self.num_of_bits:
            dataset = random_state.randint(0, 2, size=(datasets_size*2, self.num_of_bits), dtype=bool)
            dataset = numpy.unique(dataset, axis=1)
            random_state.shuffle(dataset)
            dataset = dataset[:datasets_size,:] # we oversample to compensate for the numpy.unique
        else:
            # we should create all combinationa
            dataset = numpy.array(list(itertools.product([0, 1], repeat=self.num_of_bits)), dtype=bool)
            random_state.shuffle(dataset)
        return dataset
        
    def setup(self, stage: str):
        full_dataset = self._build_dataset_held_out_at_input_level()
        self.validation_dataset = full_dataset[:self.eval_batch_size*self.eval_steps, :]
        self.test_dataset = full_dataset[self.eval_batch_size*self.eval_steps:2*self.eval_batch_size*self.eval_steps, :]
        self.training_dataset = full_dataset[2*self.eval_batch_size*self.eval_steps:, :]
        
    def train_dataloader(self):
        return DataLoader(BitSubsetParityDataset(self.training_dataset, self.subset, self.step_by_step), batch_size=self.train_batch_size, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(BitSubsetParityDataset(self.validation_dataset, self.subset, False), batch_size=self.eval_batch_size, pin_memory=True, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(BitSubsetParityDataset(self.test_dataset, self.subset, False), batch_size=self.eval_batch_size, pin_memory=True, num_workers=self.num_workers)

    @staticmethod
    def add_data_module_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--num_of_bits', type=int, default=64)
        parser.add_argument('--train_batch_size', type=int, default=32)
        parser.add_argument('--eval_batch_size', type=int, default=32)
        parser.add_argument('--eval_steps', type=int, default=32)
        parser.add_argument('--step_by_step', dest='step_by_step', action='store_true')
        parser.add_argument('--single_step', dest='step_by_step', action='store_false')
        parser.set_defaults(step_by_step=False)
        return parser


if __name__ == '__main__':
    data_module = BitSubsetParityDataModule()
    data_module.prepare_data()
    data_module.setup('fit')
    data_loader = data_module.train_dataloader()
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
