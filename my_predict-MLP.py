# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import pickle
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    # "coco": "./checkpoints/coco_prefix-015.pt",
    # "coco": "./data/checkpoints-p40-only_prefix/coco_prefix-006.pt",
    # "coco": "./data/checkpoints-full-token-length-MLP/coco_prefix-021.pt",
    "coco": "./data/checkpoints-max_tokens_76-prefix_length_35-bs_24-MLP/coco_prefix-008.pt",
    # "conceptual-captions": "conceptual_weights.pt",
}

D = torch.device
CPU = torch.device("cpu")
DEVICE = "cuda:1"
# DEVICE = "cpu"

gpt2_model = "gpt2"
        
def main(data, model_name="coco", use_beam_search=False, is_only_prefix=False):
    # Device setup
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = DEVICE
    
    # Load CLIP model and tokenizer
    # clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    
    # Model loading
    models = {}
    prefix_length = 35
    for key, weights_path in WEIGHTS_PATHS.items():
        # model = ClipCaptionModel(prefix_length, 768)
        if is_only_prefix:
            model = ClipCaptionPrefix(prefix_length, 768)
        else:
            model = ClipCaptionModel(prefix_length, 768)
        checkpoint = torch.load(weights_path, map_location=torch.device(device))
    
        # Filter out only the keys for the MLP (clip_project)
        # mlp_state_dict = {
        #     k: v for k, v in checkpoint.items() if k.startswith('clip_project')
        # }
        # mlp_state_dict = {k.replace("clip_project.", ""): v for k, v in mlp_state_dict.items()}
        model = torch.compile(model)
        model.load_state_dict(checkpoint)
        
        # Load the filtered state dict into the clip_project (MLP) module
        # model.clip_project.load_state_dict(mlp_state_dict)
        # model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model = model.eval()
        model = model.to(device)
        models[key] = model

    # Select the specified model
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    model = models[model_name]

    for i in range(len(data["captions"])):
    # Load the prompt embedding
        with torch.no_grad():
            prompt_emb = data['clip_embedding'][i].to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prompt_emb).reshape(1, prefix_length, -1)
        
        # Generate text using beam search or simple generation
        if use_beam_search:
            result = generate_beam(model, tokenizer, embed=prefix_embed)[:3]
        else:
            result = generate2(model, tokenizer, embed=prefix_embed)
        
        # Return or print the result
        print("========================================================")
        print(f"original: {data['captions'][i]['caption']}\n")
        for j, res in enumerate(result):
            print(f"generated_{j}: {res}")
    # return result


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # if prefix_length > 10:  # not enough memory
        #     self.clip_project = nn.Linear(
        #         prefix_size, self.gpt_embedding_size * prefix_length
        #     )
        # else:
        #     self.clip_project = MLP(
        #         (
        #             prefix_size,
        #             (self.gpt_embedding_size * prefix_length) // 2,
        #             self.gpt_embedding_size * prefix_length,
        #         )
        #     )
            
        self.clip_project = MLP(
            (
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            )
        )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=75,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    generated = None
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    # print(generated.shape)
    # print(embed.shape)
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    # print(tokens.shape)
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

if __name__ == "__main__":
    # Example usage: adjust the inputs as needed
    
    model_name = "coco"
    use_beam_search = True
    
    training_dataset = 1
    is_only_prefix = False
    
    if training_dataset == 1:
        data_path = "./80_100_data.pkl"
        data_path = "./80_100_data.pkl"
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        all_data["captions"] = all_data["captions"][:10]
        all_data["clip_embedding"] = all_data["clip_embedding"][:10]
    else:
        data_path = "../diffusion/result.pt"
        all_data = dict()
        all_data["clip_embedding"] = []
        all_data["clip_embedding"].append(torch.load(data_path, map_location='cpu'))
        all_data["captions"] = []
        all_data["captions"].append({"caption": "anime girl walking in the woods"})

    main(all_data, model_name, use_beam_search, False)