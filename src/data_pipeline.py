import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_local_data():
    # Use a relative path pointing to your local data folder
    file_path = os.path.join("data", "Sentences_AllAgree.txt")
    
    print(f"🔄 Loading local data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Could not find the file at {file_path}. Did you unzip it and place it in the data/ folder?")

    # The file is ISO-8859-1 encoded and uses '@' as a separator between text and label
    # Example format: "The shares rose 2% .@positive"
    df = pd.read_csv(
        file_path, 
        sep="@", 
        names=["sentence", "label_text"], 
        encoding="ISO-8859-1"
    )
    
    # Strip any accidental whitespace from text/labels
    df["sentence"] = df["sentence"].str.strip()
    df["label_text"] = df["label_text"].str.strip()
    
    # Convert text labels ('negative', 'neutral', 'positive') to numbers (0, 1, 2) to match standard ML requirements
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label_text"].map(label_map)
    
    print(f"✅ Successfully loaded {len(df)} local financial phrases.")
    print("\n📊 Class Distribution:")
    print(df["label_text"].value_counts())
    
    # Split into 80% Training and 20% Testing data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    print(f"\n📂 Local Split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
    return train_df, test_df

if __name__ == "__main__":
    train, test = load_and_prepare_local_data()
    print("\n👀 Sample local headline:")
    print(f"Text: {train.iloc[0]['sentence']}")
    print(f"Label: {train.iloc[0]['label_text']} (ID: {train.iloc[0]['label']})")