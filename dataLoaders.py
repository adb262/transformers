from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):
    """Dataset for the symptom-disease csv."""

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, encoding='utf-8', error_bad_lines=False, engine='python').reset_index()
        self.df.EVIDENCES = self.df.EVIDENCES.apply(eval).str.join(" ")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        symptoms = self.df.loc[idx].EVIDENCES
        labels = self.df.loc[idx].PATHOLOGY
        return labels, symptoms