import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import pickle

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        print(f"max length: {self.tokenizer.model_max_length}")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        # self.image_ids = [caption["image_id"] for caption in captions_raw] #
        self.captions = [caption['caption'] for caption in captions_raw]

        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        for caption in captions_raw:
            self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./training_data.pkl')
parser.add_argument('--out_dir', default='./data/checkpoints')
parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--prefix_length', type=int, default=10)
parser.add_argument('--prefix_length_clip', type=int, default=10)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--is_rn', dest='is_rn', action='store_true')
parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
args = parser.parse_args()
prefix_length = args.prefix_length
dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

print(f"Dataset size: {len(dataset)}")
print(f"normalize_prefix: {args.normalize_prefix}")
print(len(train_dataloader))
