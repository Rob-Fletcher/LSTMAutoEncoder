import tables
import torch
from torch.utils.data import Dataset
from random import randint, seed
from tqdm import tqdm

class PathData(Dataset):
    """ Load path data as vector """

    def __init__(self, data_file, sequence_length=100):
        """Initialize the class to hold the paths

        TODO: will need to make this so it can read in multiple data files
        currently can only read one at a time.
        """
        seed()
        self.data_file = data_file
        self.seq_len = sequence_length
        self.detections = tables.open_file(data_file).root.detections
        uniqueIDs_all = torch.tensor(self.detections[:]['ID'], dtype=torch.int32).unique()
        self.mean = torch.tensor([self.detections[:]['x1'].mean(), self.detections[:]['y1'].mean()], dtype=torch.float64)
        self.std = torch.tensor([self.detections[:]['x1'].std(), self.detections[:]['y1'].std()], dtype=torch.float64)
        self.min = torch.tensor([self.detections[:]['x1'].min(), self.detections[:]['y1'].min()], dtype=torch.float64)
        self.max = torch.tensor([self.detections[:]['x1'].max(), self.detections[:]['y1'].max()], dtype=torch.float64)
        uniqueList = []
        print("Removing all sequences with a length less than 10.")
        for id in tqdm(uniqueIDs_all):
            path = self.detections.read_where("""ID=={}""".format(id))
            if len(path) <= 10:
                continue
            else:
                uniqueList.append(id)

        self.uniqueIDs = torch.tensor(uniqueList, dtype=torch.int32)
        print(f"Total paths in dataset: {self.uniqueIDs.shape[0]}")


    def __len__(self):
        """ Get number of paths from the h5 file """
        return len(self.uniqueIDs)

    def __getitem__(self, idx):
        path = self.detections.read_where("""ID=={}""".format(self.uniqueIDs[idx]))[['x1','y1']]
        path = torch.tensor(path.tolist(), dtype=torch.float64)
        #path.sub_(self.mean)
        #path.div_(self.std)
        path.sub_(self.min)
        path.div_(self.max - self.min)
        subPath = torch.zeros([self.seq_len,2], dtype=torch.float32)
        length = len(path)
        if length > self.seq_len:
            startPoint = randint(0, length-self.seq_len)
            subPath[:] = path[startPoint:startPoint+self.seq_len]
        else:
            #print(f"length less than {self.seq_len}")
            subPath[:length] = path[:]
        return subPath
