# src/dataset_factory.py
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_pipeline import load_and_prepare_local_data

class FinancialPhraseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Format tokens and labels into a clean dictionary for our model
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def verify_factory():
    print("🛠️ Testing Dataset Factory...")
    train_df, _ = load_and_prepare_local_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize
    encodings = tokenizer(train_df["sentence"].tolist(), truncation=True, padding=True, max_length=64)
    labels = train_df["label"].tolist()
    
    # Create dataset object
    dataset = FinancialPhraseDataset(encodings, labels)
    print(f"✅ Dataset successfully created! Total training samples: {len(dataset)}")
    
    # Test fetch the very first sample object
    first_sample = dataset[0]
    print(f"📦 Sample 0 keys parsed: {list(first_sample.keys())}")

if __name__ == "__main__":
    verify_factory()