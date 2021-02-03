"""
Spike raster generation from a poisson distribution
====================================================

"""
from torch.utils import data
from pathlib import Path
import torch


class SpikeRaster(data.Dataset):
    """

    """

    def __init__(self, fname=None, list_of_ids=None):
        """

        Class for efficient loading of saved spike data.\n
        :param fname: Location of saved spike raster.
        :param list_of_ids: List of time points.
        """
        if fname is None:
            fname = Path.cwd() / 'snnpytorch' / 'data' / 'input_data.pt'

        # Load saved data
        self.data = torch.load(fname)

        if list_of_ids is None:
            list_of_ids = list(range(len(self.data)))
        self.list_of_ids = list_of_ids

    def __len__(self) -> int:
        return len(self.list_of_ids)

    def __getitem__(self, index):
        return self.data[self.list_of_ids[index], :]
