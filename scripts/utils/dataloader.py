from torch.utils.data import DataLoader
import itertools

class DataLoaderX(object):
    def __init__(self, dataset1, dataset2, batch_size, worker_init_fn, rate=1):
        self.rate = rate

        self.dataloader1 = DataLoader(dataset1, num_workers=2, batch_size=batch_size, worker_init_fn=worker_init_fn)
        self.dataloader2 = DataLoader(dataset2, num_workers=2, batch_size=batch_size, worker_init_fn=worker_init_fn)

        self.iterator = itertools.chain.from_iterable(iter([self.dataloader1] + [self.dataloader2] * self.rate))

    def __len__(self):
        return len(self.dataloader1) + len(self.dataloader2) * self.rate

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = itertools.chain.from_iterable(iter([self.dataloader1] + [self.dataloader2] * self.rate))
            raise StopIteration

    def __iter__(self):
        return self


