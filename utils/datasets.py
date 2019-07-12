import tables
import torch
from torch.utils.data import Dataset

class PathData(Dataset):
    """ Load path data as vector """

    def __init__(self, data_file):
        """Initialize the class to hold the paths

        TODO: will need to make this so it can read in multiple data files
        currently can only read one at a time.
        """
        self.data_file = data_file
        self.detections = tables.open_file(data_file).root.detections
        self.uniqueIDs = torch.tensor(self.detections[:]['ID'], dtype=torch.int32).unique()
        self.mean = torch.tensor([self.detections[:]['x1'].mean(), self.detections[:]['y1'].mean()], dtype=torch.float64)


    def __len__(self):
        """ Get number of paths from the h5 file """
        return len(self.uniqueIDs)

    def __getitem__(self, idx):
        path = self.detections.read_where("""ID=={}""".format(self.uniqueIDs[idx]))[['x1','y1']]
        path = torch.tensor(path.tolist(), dtype=torch.float64)
        path.sub_(self.mean)
        return path.float()
