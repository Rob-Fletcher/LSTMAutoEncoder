import tables
import torch
from torch.utils.data import Dataset
from random import randint, seed

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
        uniqueList = []
        for id in uniqueIDs_all:
            path = self.detections.read_where("""ID=={}""".format(id))
            if len(path) < 10:
                print(f"ID: {id} has squence length less than 10. Discarding")
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
        path.sub_(self.mean)
        subPath = torch.zeros([100,2], dtype=torch.float64)
        length = len(path)
        if length > 100:
            startPoint = randint(0, length-100)
            subPath[:] = path[startPoint:startPoint+100]
        return path
