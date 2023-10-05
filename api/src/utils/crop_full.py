import os
import uuid

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.videoio import save_video_with_watermark


def crop_full(full_video_path, crop_info, new_audio_path, av_path):

    rate = 8
    rate_ = 5

    if not os.path.isfile(full_video_path):
        print("ko co video")

    video_stream = cv2.VideoCapture(full_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(frame)

    h, w = full_frames[0].shape[:2]
# ---------------------------------------------------------------------------------------
    clx, cly, crx, cry = crop_info[1]
    crx = int(crx + min(clx, w-crx)/rate)
    clx = int(clx - min(clx, w-crx)/rate)
    cly = int(cly - min(cly, h-cry)/rate_)
    cry = min(int(cry + (cry - cly)/rate_), h)

    new_w = crx - clx
    new_h = cry - cly
    tmp_path = str(uuid.uuid4()) + ".mp4"
    out_tmp = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (new_w, new_h)
    )
    for crop_frame in tqdm(full_frames, "cropFromFull:"):

        crop_frame = crop_frame[cly:cry, clx:crx]

        out_tmp.write(crop_frame)

    out_tmp.release()

    save_video_with_watermark(
        tmp_path, new_audio_path, av_path, watermark=False
    )
