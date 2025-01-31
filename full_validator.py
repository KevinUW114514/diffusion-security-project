import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import clip
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 114514
torch.manual_seed(random_seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(random_seed)
 
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class ClipWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model  # the original CLIP model

    def forward(self, tokens):
        # Internally call CLIP’s encode_text
        return self.clip_model.encode_text(tokens)

def main():
    # path = "clip_text_normalized_embeddings_checkpoint.pt"
    # prompt_name = "prompts"
    # embeddings_name = "normalized_embeddings"
    
    path = "filtered_clip_text_normalized_embeddings_checkpoint.pt"
    prompt_name = "filtered_prompts"
    embeddings_name = "filtered_embeddings"
    
    checkpoint_data = torch.load(path)
    pending_prompts = checkpoint_data[prompt_name]
    pending_embeddings = checkpoint_data[embeddings_name]
    
    batch_size = 10000

    # Load CLIP on the main device (cuda:0)
    base_clip_model, _ = clip.load("ViT-L/14", device=device)
    model = ClipWrapper(base_clip_model).to(device)

    # Wrap with DataParallel to use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Use both GPUs [0,1]
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # Prepare DataLoader
    dataset = TextDataset(pending_prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    embeddings = []

    with torch.no_grad():
        for texts in tqdm(dataloader, desc="Processing Batches"):
            
            # Tokenize & move inputs to the same main device
            tokens = clip.tokenize(texts, truncate=True).to(device)

            # Forward pass through the model
            #    With DataParallel, you can call encode_text directly
            #    and it will be automatically parallelized.
            text_features = model(tokens)

            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Move to CPU to store or concatenate
            embeddings.append(text_features.cpu())

    # Concatenate all embeddings and save
    normalized_embeddings = torch.cat(embeddings)
    print(f"Check: {torch.allclose(normalized_embeddings, pending_embeddings, atol=1e-2)}")

if __name__ == "__main__":
    main()
