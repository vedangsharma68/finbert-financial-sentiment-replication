# src/model_pipeline.py
import torch
from transformers import BertConfig, BertForSequenceClassification

def initialize_finbert_model():
    print("\n⚙️ Configuring model architecture...")
    
    # 1. Define the custom configuration
    # We use bert-base-uncased as the foundation, but specify 3 output labels
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=3, 
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )
    
    print("🤖 Initializing BertForSequenceClassification weights blueprint...")
    # 2. Load the model skeleton with our custom financial head
    # This might show a warning about weights being newly initialized—that is normal and expected!
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        config=config
    )
    
    print("✅ Model structure initialized successfully!")
    
    # Print the model summary/architecture to verify the final classification layer
    print(f"\n🔍 Classification Head Structure:\n{model.classifier}")
    
    return model

if __name__ == "__main__":
    initialize_finbert_model()