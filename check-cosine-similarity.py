import torch
from tqdm import tqdm
import pickle
import clip

data = [
    {
        "prompt": "anime visual, illustrated portrait of a young astronaut girl floating in space, colorful, cute face by ilya kuvshinov, yoshinari yoh, makoto shinkai, katsura masakazu, dynamic pose, ( ( mads berg ) ), kyoani, rounded eyes, crisp and sharp, cel shad, anime poster, halftone shading, christopher balaskas, dithered",
        "generated": "anime portrait of a cute young astronaut girl floating in space, mandelbulb, ilya kuvshinov, jamie hewlett, ilya kuvshinov, matte, gradation, bold shapes, hard edges, studio lighting, film noir, rich vivid colors, ambient lighting,"
    },
    {
        "prompt": "greed deadly sin represented by a beautiful woman surrounded by gold, jewels, and treasure, style of peter mohrbacher, vray, highly detailed, luxury, fractal, golden ratio, elegant, epic",
        "generated": "sorceress surrounded by gold, jewels, gold, peter mohrbacher style, golden ratio, rule of thirds, insanely detailed and intricate, hypermaximalist, elegant, ornate, luxury, elite, matte painting, cinematic, cgsociety"
    },
    {
        "prompt": "beautiful shimmering liquid creatures, in a shimmering cave, silver details, bioluminescent, translucent, extreme close - up, macro, concept art, intricate, detailed, award - winning, cinematic, octane render, 8 k, photorealistic, by emil melmoth, by wayne barlowe, by francis bacon, by jean - michel basquiat, by gustave moreau",
        "generated": "bioluminescent creatures, translucent, shimmering, 8 k, octane render, close - up, intricate, elegant, dramatic lighting, emotionally evoking symbolic metaphor, highly detailed, lifelike, photorealistic, cinematic lighting, art, very coherent, hyper realism, high detail, octane render, 8 k"
    }
]

data = [
    {
        "prompt": "anime visual, illustrated portrait of a young astronaut girl floating in space, colorful, cute face by ilya kuvshinov, yoshinari yoh, makoto shinkai, katsura masakazu, dynamic pose, ( ( mads berg ) ), kyoani, rounded eyes, crisp and sharp, cel shad, anime poster, halftone shading, christopher balaskas, dithered",
        "generated": "anime portrait of a cute young astronaut girl floating in space, mandelbulb, ilya kuvshinov, jamie hewlett, ilya kuvshinov, matte, gradation, bold shapes, hard edges, studio lighting, film noir, rich vivid colors, ambient lighting,"
    },
    {
        "prompt": "greed deadly sin represented by a beautiful woman surrounded by gold, jewels, and treasure, style of peter mohrbacher, vray, highly detailed, luxury, fractal, golden ratio, elegant, epic",
        "generated": "sorceress surrounded by gold, jewels, gold, peter mohrbacher style, golden ratio, rule of thirds, insanely detailed and intricate, hypermaximalist, elegant, ornate, luxury, elite, matte painting, cinematic, cgsociety"
    },
    {
        "prompt": "beautiful shimmering liquid creatures, in a shimmering cave, silver details, bioluminescent, translucent, extreme close - up, macro, concept art, intricate, detailed, award - winning, cinematic, octane render, 8 k, photorealistic, by emil melmoth, by wayne barlowe, by francis bacon, by jean - michel basquiat, by gustave moreau",
        "generated": "bioluminescent creatures, translucent, shimmering, 8 k, octane render, close - up, intricate, elegant, dramatic lighting, emotionally evoking symbolic metaphor, highly detailed, lifelike, photorealistic, cinematic lighting, art, very coherent, hyper realism, high detail, octane render, 8 k"
    }
]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

clip_model, _ = clip.load("ViT-L/14", device=device)

for sample in data:
    prompt_embedding = clip_model.encode_text(clip.tokenize(sample["prompt"], truncate=True).to(device))
    generated_embedding = clip_model.encode_text(clip.tokenize(sample["generated"], truncate=True).to(device))

    similarity = torch.nn.functional.cosine_similarity(prompt_embedding, generated_embedding, dim=-1)
    print("=" * 80)
    print(similarity.item())