import torch
from diffusers import StableDiffusionPipeline

model_id = "segmind/portrait-finetuned"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

pipe.save_pretrained("stable-diffusion-v1-4")
print("Successful!")
