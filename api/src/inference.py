import os
import shutil
import sys
import uuid

import torch
from pydub import AudioSegment

from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from src.check_crop import Image_Preprocess
from rembg import remove


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker:
    def __init__(
        self, checkpoint_path="checkpoints", config_path="src/config", lazy_load=False
    ):
        if torch.cuda.is_available():
            device = "cuda:0"
            print("cudaaaa")
        else:
            device = "cpu"
            print("cpuuuuuuuuu")

        self.device = device

        os.environ["TORCH_HOME"] = checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    def test(
        self,
        source_image,
        driven_audio,
        preprocess="full",
        still_mode=False,
        use_enhancer=False,
        batch_size=1,
        size=256,
        pose_style=10,
        exp_scale=1.0,
        use_idle_mode=False,
        length_of_audio=0,
        use_blink=True,
        result_dir="./results/",
    ):
        self.sadtalker_paths = init_path(
            self.checkpoint_path, self.config_path, size, False, preprocess
        )
        print(self.sadtalker_paths)

        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(
            self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(
            self.sadtalker_paths, self.device)
        self.img_pre = Image_Preprocess(self.device)

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)
        input_dir = os.path.join(save_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        pic_path = os.path.join(input_dir, os.path.basename(source_image))
        pic_path_source = os.path.join(
            input_dir, os.path.basename(source_image))

        shutil.copy(source_image, input_dir)

        # audio_path
        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(
                input_dir, os.path.basename(driven_audio))

            # mp3 to wav
            if ".mp3" in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace(
                    ".mp3", ".wav"), 16000)
                audio_path = audio_path.replace(".mp3", ".wav")
            else:
                shutil.copy(driven_audio, input_dir)

        if not still_mode and preprocess == 'crop':
            source_image = self.img_pre.img_pre(source_image)
            pic_path_source = source_image
            pic_name = os.path.splitext(os.path.split(source_image)[-1])[0]

            output_path = './' + pic_name + '_nobg.png'

            with open(source_image, 'rb') as i:
                with open(output_path, 'wb') as o:
                    input = i.read()
                    output = remove(input)
                    o.write(output)

            pic_path = os.path.join(input_dir, os.path.basename(output_path))

            shutil.copy(output_path, input_dir)

        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)

        first_coeff_path, crop_pic_path, crop_info, pic_path_full = self.preprocess_model.generate(
            pic_path, first_frame_dir, preprocess, True, size, still_mode=still_mode
        )

        batch = get_data(
            first_coeff_path,
            audio_path,
            self.device,
            ref_eyeblink_coeff_path=None,
            still=still_mode,
            idlemode=use_idle_mode,
            length_of_audio=length_of_audio,
            use_blink=use_blink,
        )
        coeff_path = self.audio_to_coeff.generate(
            batch, save_dir, pose_style, ref_pose_coeff_path=None
        )

        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            preprocess=preprocess,
            size=size,
            expression_scale=exp_scale,
            pic_path_full=pic_path_full
        )

        return_path = self.animate_from_coeff.generate(data, save_dir,  pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None,
                                                       preprocess=preprocess, img_size=size, pic_path_source=pic_path_source, still_mode=still_mode)

        return return_path
