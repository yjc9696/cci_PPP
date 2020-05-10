from torch.utils.data import Dataset, DataLoader


class TrainSet(Dataset):
    def __init__(self, dataset, score=None):
        self.dataset = dataset

        self.score = score
        if self.score is not None:
            assert len(score) == len(dataset), 'error'
            pass

    def __getitem__(self, index):
        type1, type2, label, score,  _, _ = self.dataset[index]
        if self.score is not None:
            score = self.score[index]
        return type1, type2, label, score

    def __len__(self):
        return len(self.dataset)

