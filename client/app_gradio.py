import io
import json

import gradio as gr
import requests
from PIL import Image
import os


def clear():
    return None


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def validate_prompt(prompt):
    if len(prompt) > 0:
        return None
    raise gr.Error("You must input text to generate image")


def validate_video_input(audio, image):
    if audio is None:
        raise gr.Error("Please upload your audio")
    elif image is None:
        raise gr.Error("Please upload or generate image")
    return None


def enable_state_video():
    return {
        file_output: gr.update(interactive=True),
        result_image: gr.update(interactive=True),
        run_button: gr.update(interactive=True),
        prompt: gr.update(interactive=True),
        video_button: gr.update(interactive=True)
    }


def disable_state_video():
    return {
        file_output: gr.update(interactive=False),
        result_image: gr.update(interactive=False),
        run_button: gr.update(interactive=False),
        prompt: gr.update(interactive=False),
        video_button: gr.update(interactive=False)
    }


def enable_state_image():
    return {
        file_output: gr.update(interactive=True),
        run_button: gr.update(interactive=True),
        prompt: gr.update(interactive=True),
        video_button: gr.update(interactive=True)
    }


def disable_state_image():
    return {
        file_output: gr.update(interactive=False),
        run_button: gr.update(interactive=False),
        prompt: gr.update(interactive=False),
        video_button: gr.update(interactive=False)
    }


def generate_image(prompt):
    res = requests.post("http://model_api:8000/prediction", data={"prompt": prompt})

    byte = res.content

    img = Image.open(io.BytesIO(byte))
    return img

def generate_video(audio, image, still_mode):
    files = [('files', open(audio, 'rb')), ('files', open(image, 'rb'))]
    res = requests.post(
        "http://model_api:8000/video",
        files=files,
        data={"still_mode": still_mode}
    ) # still_mode is boolean
    if res.status_code == 200 or res.status_code == 307:
        byte = res.content
        with open("result.mp4", 'wb') as f:
            f.write(byte)
        return "result.mp4"

    raise gr.Error("Cannot detect face")


with gr.Blocks() as demo:
    gr.Markdown(
        "<div align='center'> <h1>Video Synthesis from Portrait Image and Audio </h1> \
    <hr></div>"
    )
    with gr.Row():
        with gr.Column() as input_col:
            gr.Markdown(
                "<div align='center'><h2>Input Image and Audio</h2></div>")
            file_output = gr.Audio(source="upload", type="filepath")

            gr.Markdown(
                "### Choose 1 of 2 options: Upload image or Generate image by Stable Diffusion")

            prompt = gr.Textbox(label="Prompt", placeholder="")
            run_button = gr.Button("Generate Image")
            result_image = gr.Image(
                label="Result", show_label=False, type="filepath", width=742, height=450)

            gr.Markdown("### Text Examples")
            gr1 = gr.Examples(
                ["a photo of a woman with pink long hair, oval face, young, white, talking"],
                prompt,
                result_image,
            )

            gr.Markdown("### Image Examples")
            gr2 = gr.Examples(
                examples=[os.path.join(os.path.dirname(__file__), "example/example_img.png"), 
                            os.path.join(os.path.dirname(__file__), "example/example_img2.png"),
                            os.path.join(os.path.dirname(__file__), "example/example_img3.png")],
                inputs=result_image,
                outputs=result_image,
            )

        with gr.Column():          
            gr.Markdown("<div align='center'><h2>Video Synthesis Result</h2></div>")
            still_mode = gr.Checkbox(label="still mode", min_width=80)
            video_button = gr.Button("Generate Video")
            result_video = gr.Video(label="Generated Video")

    run_button.click(
        validate_prompt,
        inputs=[prompt],
        outputs=[]
    ).then(
        disable_state_image,
        inputs=[],
        outputs=[file_output, run_button, prompt, video_button]
    ).success(
        generate_image,
        inputs=[prompt],
        outputs=[result_image]
    ).then(
        enable_state_image,
        inputs=[],
        outputs=[file_output, run_button, prompt, video_button]
    )

    video_button.click(
        validate_video_input,
        inputs=[file_output, result_image],
        outputs=[],
    ).then(
        disable_state_video,
        inputs=[],
        outputs=[file_output, result_image, run_button, prompt, video_button]
    ).success(
        generate_video,
        inputs=[file_output, result_image, still_mode],
        outputs=[result_video],
    ).then(
        enable_state_video,
        inputs=[],
        outputs=[file_output, result_image, run_button, prompt, video_button]
    )

demo.launch(server_name="0.0.0.0")
