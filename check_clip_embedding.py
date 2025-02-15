import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import clip
from tqdm import tqdm
import numpy as np
import sys

device = torch.device("cpu")

random_seed = 114514
torch.manual_seed(random_seed)
 
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        if 'prompt' not in self.data.columns:
            raise ValueError("Dataset must contain a 'prompt' column.")
        
        self.data = self.data['prompt'].astype(str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class ClipWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model 

    def forward(self, tokens):
        # Internally call CLIP’s encode_text
        return self.clip_model.encode_text(tokens)

def main():
    dataset_path = "./data/diffusiondb_0_60.csv"
    batch_size = 10

    # Load CLIP on the main device (cuda:0)
    base_clip_model, _ = clip.load("ViT-L/14", device=device)
    model = ClipWrapper(base_clip_model).to(device)

    # Prepare DataLoader
    dataset = TextDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    embeddings = []
    
    with torch.no_grad():
        for texts in tqdm(dataloader, desc="Processing Batches"):
            
            # Tokenize & move inputs to the same main device
            tokens = clip.tokenize(texts, truncate=True).to(device)

            # Forward pass through the model
            text_features = model(tokens)
            print(f"Shape: {text_features.shape}")
            break

    # Concatenate all embeddings and save
    print("nao")

if __name__ == "__main__":
    main()
