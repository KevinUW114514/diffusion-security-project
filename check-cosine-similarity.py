import torch
from tqdm import tqdm
import pickle
import clip

data = [
    # {
    #     "prompt": "beautiful shimmering liquid creatures, in a shimmering cave, silver details, bioluminescent, translucent, extreme close - up, macro, concept art, intricate, detailed, award - winning, cinematic, octane render, 8 k, photorealistic, by emil melmoth, by wayne barlowe, by francis bacon, by jean - michel basquiat, by gustave moreau",
    #     "generated": "bioluminescent translucent creatures, close - up, intricate details, cinematic, 8 k, octane render, volumetric lighting, vivid, beautiful, hyperrealistic, polished, micro details, 3 d sculpture, structure, ray trace, masterpiece, award - winning, trending, featured, feng zhu, shaddy safadi, noah bradley, tyler edlin, jordan grimmer, darek zabrocki, neil blevins, tuomas korpi  trending on artstation, artstationhd, artstationhq, unreal engine, 4 k, 8 k"
    # },
    # {
    #     "prompt": "anime visual, illustrated portrait of a young astronaut girl floating in space, colorful, cute face by ilya kuvshinov, yoshinari yoh, makoto shinkai, katsura masakazu, dynamic pose, ( ( mads berg ) ), kyoani, rounded eyes, crisp and sharp, cel shad, anime poster, halftone shading, christopher balaskas, dithered",
    #     "generated": "anime visual portrait of a cute young astronaut girl floating in space, ilya kuvshinov, ilya kuvshinov, murata range, pixiv, cel shading, vivid colors, matte print, bold shapes, hard edges, studio ghibli!!!, anime illustration, anime key visual"
    # },
    # {
    #     "prompt": "happy and cute face of jack russel terrier, face only, smile, pencil drawing, pastel, by marc simonetti",
    #     "generated": "face of a happy jack russel terrier, full face, small nose, pencil drawing, colored, sketch, by Marc Simonetti"
    # },
    # {
    #     "prompt": "an abolsutely cute and gorgeous 2 0 year old woman, photo selfie, realistic, highly detailed",
    #     "generated": "a cute 2 2 year old woman selfie, realistic, highly detailed, cute   selfie"
    # }
    {
        "prompt": "A dog running in a field",
        "generated": "A horse sprinting through grass"
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