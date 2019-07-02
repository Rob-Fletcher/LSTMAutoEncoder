import tables
import torch
from torch.utils.data import Dataset

class PathData(Dataset):
    """ Load path data as vector """

    def __init__(self, data_file):
        self.data_file = data_file
        self.detections = tables.open_file(data_file).root.detections
        self.uniqueIDs = self.detections['ID'].unique


    def __len__(self):
        """ Get number of paths from the h5 file """
        return len()
