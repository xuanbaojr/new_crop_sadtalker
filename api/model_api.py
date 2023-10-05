import io
import os
import time
from contextlib import asynccontextmanager
from typing import List

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response

from src.inference import SadTalker


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck" 
    models["negative_prompt"] = negative_prompt

    if torch.cuda.is_available():
        device = 'cuda:0'
        dtype = torch.float16
    else:
        device = 'cpu'
        dtype = torch.float32
    model_id = "portrait-finetuned"

    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe.enable_attention_slicing()
    pipe.to(device)
    pipe.unet.to(device=device, dtype=dtype, memory_format=torch.channels_last)

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


@app.post("/prediction/")
@torch.inference_mode()
def predict(prompt: str = Form(...)):
    image = models["pipe"](
        prompt=prompt, negative_prompt=models["negative_prompt"], 
        width=512, height=512, num_inference_steps=25, guidance_scale = 7.5, num_images_per_prompt=1
    ).images[0]
    img = io.BytesIO()
    image.save(img, format="PNG")
    img_byte_arr = img.getvalue()
    return Response(content=img_byte_arr, media_type="image/png")


@app.post("/video/")
def create_item(still_mode: bool = Form(...), files: List[UploadFile] = File(...)):
    audio_path, image_path = files[0].filename, files[1].filename

    for file in files:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
        file.file.close()

    result_path = models["sad_talker"].test(image_path, audio_path, still_mode=still_mode)

    with open(result_path, 'rb') as fd:
        contents = fd.read()

    return Response(content=contents, media_type="video/mp4")
