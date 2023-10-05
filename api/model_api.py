import io
import os
import time
from contextlib import asynccontextmanager
from typing import List


from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response

from src.inference import SadTalker
import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from torch import Generator


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = 'segmind/small-sd'  # Path to the appropriate model-type
    # Insert your prompt below.
    prompt = "Faceshot Portrait of pretty young (18-year-old) Caucasian wearing a high neck sweater, (masterpiece, extremely detailed skin, photorealistic, heavy shadow, dramatic and cinematic lighting, key light, fill light), sharp focus, BREAK epicrealism"
    # Insert negative prompt below. We recommend using this negative prompt for best results.
    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

# Below code will run on gpu, please pass cpu everywhere as the device and set 'dtype' to torch.float32 for cpu inference.
    with torch.inference_mode():
        gen = Generator("cuda:0")
        gen.manual_seed(1674753452)
        pipe = DiffusionPipeline.from_pretrained(
            path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        pipe.to('cuda:0')
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        pipe.unet.to(device='cuda:0', dtype=torch.float16,
                     memory_format=torch.channels_last)

        img = pipe(prompt=prompt, negative_prompt=negative_prompt, width=512, height=512,
                   num_inference_steps=25, guidance_scale=7, num_images_per_prompt=1, generator=gen).images[0]
        img.save("image01.png")

        models["pipe"] = pipe

        sad_talker = SadTalker(
            checkpoint_path="checkpoints", config_path="src/config", lazy_load=True
        )

        models["sad_talker"] = sad_talker

    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post("/video/")
@torch.inference_mode()
def predict(prompt: str = Form(...)):

    print("xysas")
    image = models["pipe"](
        prompt=prompt, negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        width=512, height=512, num_inference_steps=25, guidance_scale=7.5, num_images_per_prompt=1
    ).images[0]

    img = io.BytesIO()
    image.save(img, format="PNG")
    img_byte_arr = img.getvalue()
    return Response(content=img_byte_arr, media_type="image/png")


@app.post("/prediction/")
def create_item(still_mode: bool = Form(...), crop: bool = Form(...), files: List[UploadFile] = File(...)):
    audio_path, image_path = files[0].filename, files[1].filename

    for file in files:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
        file.file.close()
    if crop:
        preprocess = "crop"
    else:
        preprocess = "full"

    result_path = models["sad_talker"].test(
        image_path, audio_path, preprocess=preprocess, still_mode=still_mode)

    with open(result_path, 'rb') as fd:
        contents = fd.read()

    return Response(content=contents, media_type="video/mp4")
