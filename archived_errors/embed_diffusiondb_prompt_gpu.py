import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import clip
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        if 'prompt' not in self.data.columns:
            raise ValueError("Dataset must contain a 'prompt' column.")
        self.data['prompt'] = [str(x) for x in self.data['prompt']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]['prompt']
    
def tokenize_clip(text):
    return clip.tokenize(text, truncate=True)

def main():
    dataset_path = "diffusiondb.csv"  # Path to your large dataset
    batch_size = 10000


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-L/14", device=device)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     # Use both GPUs [0,1]
    #     model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # Prepare DataLoader
    dataset = TextDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    embeddings = []

    with torch.no_grad():
        for texts in tqdm(dataloader, desc="Processing Batches"):
                
            tokens = clip.tokenize(texts, truncate=True).to(device)

            text_features = model.encode_text(tokens)

            # 6. Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 7. Move to CPU to store or concatenate
            embeddings.append(text_features.cpu())

    # Concatenate all embeddings and save
    normalized_embeddings = torch.cat(embeddings)
    torch.save({"normalized_embeddings": normalized_embeddings},
               "clip_text_normalized_embeddings_checkpoint1.pt")
    print("Saved embeddings to clip_text_normalized_embeddings_checkpoint1.pt")

if __name__ == "__main__":
    main()
