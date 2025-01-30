import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import clip
from tqdm import tqdm

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
    
class ClipWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model  # the original CLIP model

    def forward(self, tokens):
        # Internally call CLIP’s encode_text
        return self.clip_model.encode_text(tokens)
    
def tokenize_clip(text_batch):
    return clip.tokenize(text_batch, truncate=True)

def main():
    dataset_path = "diffusiondb.csv"  # Path to your large dataset
    batch_size = 10000

    # 1. Decide on device (prefer "cuda:0" if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Load CLIP on the main device (cuda:0)
    base_clip_model, _ = clip.load("ViT-L/14", device=device)
    model = ClipWrapper(base_clip_model).to(device)

    # 3. Wrap with DataParallel to use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Use both GPUs [0,1]
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # Prepare DataLoader
    dataset = TextDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    embeddings = []

    with torch.no_grad():
        for texts in tqdm(dataloader, desc="Processing Batches"):
            # with ThreadPoolExecutor(max_workers=8) as executor:
            #     tokens = list(executor.map(tokenize_clip, texts))
                
            # tokens = torch.cat(tokens).to(device)
            
            # 4. Tokenize & move inputs to the same main device
            tokens = clip.tokenize(texts, truncate=True).to(device)

            # 5. Forward pass through the model
            #    With DataParallel, you can call encode_text directly
            #    and it will be automatically parallelized.
            text_features = model(tokens)

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
