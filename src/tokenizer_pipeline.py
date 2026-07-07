import torch
from transformers import BertTokenizer
from data_pipeline import load_and_prepare_local_data

def tokenize_dataset():
    # 1. Load the data using your previous script
    train_df, test_df = load_and_prepare_local_data()
    
    print("\n🔄 Initializing standard BERT tokenizer...")
    # FinBERT is built on 'bert-base-uncased', so we use its matching tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    print("🔤 Tokenizing training text sequences...")
    
    # We convert text columns into lists
    train_texts = train_df["sentence"].tolist()
    train_labels = train_df["label"].tolist()
    
    # Tokenize the text. This handles padding short sentences and truncation of long ones.
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding=True, 
        max_length=64, # Financial headlines are short, so 64 tokens is plenty
        return_tensors="pt" # Return PyTorch Tensors
    )
    
    print("✅ Tokenization successful!")
    print(f"Shape of Input IDs tensor: {train_encodings['input_ids'].shape}")
    
    # Let's inspect what the first sentence looks like converted to numeric vector IDs
    print("\n👀 Word-to-Number Representation of Sample 1:")
    print(train_encodings['input_ids'][0])
    
    return train_encodings, train_labels

if __name__ == "__main__":
    tokenize_dataset()