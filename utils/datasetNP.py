import torch
from torch.utils.data import Dataset
from random import randint, seed
import numpy as np
import os

class PathDataNP(Dataset):
    """ Load path data as vector """

    def __init__(self, data_dir, seq_len=50, pred_len=50):
        """Initialize the class to hold the paths

        TODO: will need to make this so it can read in multiple data files
        currently can only read one at a time.
        """
        seed()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.files = []
        self.paths = []

        # Find all .npy files in the data path
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.files.append(os.path.join(root, file))
                    self.paths.append(np.load(os.path.join(root, file)))

        print(f"Total files found: {len(self.files)}.")
        print(f"Total files loaded into memory {len(self.paths)}")


    def __len__(self):
        """ Get number of paths from the h5 file """
        return len(self.files)*100

    def __getitem__(self, idx):
        # path = np.load(self.files[idx%len(self.files)])
        path = self.paths[idx%len(self.paths)]
        if np.isnan(path).any():
            print("Found NaN!!!!")
        start_index = randint(0, len(path)-(self.seq_len+self.pred_len))
        subpath = torch.from_numpy(path[start_index:(start_index+self.seq_len)]).float()
        predpath = torch.from_numpy(path[(start_index+self.seq_len) : (start_index+self.seq_len+self.pred_len)]).float()
        subpath[:,:2] = subpath[:,:2].mul(10000)
        predpath[:,:2] = predpath[:,:2].mul(10000)
        return subpath, predpath
