from torch.utils.data import Dataset, DataLoader


class TrainSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        type1, type2, label, _, _ = self.dataset[index]
        
        return type1, type2, label

    def __len__(self):
        return len(self.dataset)

