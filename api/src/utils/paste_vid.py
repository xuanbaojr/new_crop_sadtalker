import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid

from src.utils.videoio import save_video_with_watermark


def paste_vid(head_video, body_video, crop_info, new_audio_path, full_video_path, body_h, body_w):

    if not os.path.isfile(head_video):
        print('file is not exist!')

    video_head = cv2.VideoCapture(head_video)
    fps = 25
    full_frame_head = []
    while 1:
        still_reading, frame = video_head.read()
        if not still_reading:
            video_head.release()
            break
        full_frame_head.append(frame)

    video_body = cv2.VideoCapture(body_video)
    fps = 25
    full_frame_body = []
    while 1:
        still_reading, frame = video_body.read()
        if not still_reading:
            video_body.release()
            break
        full_frame_body.append(frame)

    clx, cly, crx, cry = crop_info[1]

    frame_w = 256
    frame_h = 256

    clx, cly, crx, cry = crop_info[1]
    lx, ly, rx, ry = crop_info[2]
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    ox1 = int(ox1*(frame_w/body_w))
    ox2 = int(ox2*(frame_w/body_w))
    oy1 = int(oy1*(frame_h/body_h))
    oy2 = int(oy2*(frame_h/body_h))

    tmp_path = str(uuid.uuid4())+'.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(
        *'MP4V'), fps, (frame_w, frame_h))
    for key in tqdm(range(len(full_frame_head)), 'Collecting video: '):
        head = cv2.resize(full_frame_head[key].astype(
            np.uint8), (ox2 - ox1, oy2 - oy1))
        mask = 255*np.ones(head.shape, head.dtype)
        location = ((ox1+ox2)//2, (oy1 + oy2) // 2)
        gen_img = cv2.seamlessClone(
            head, full_frame_body[key], mask, location, cv2.NORMAL_CLONE)
        out_tmp.write(gen_img)

    out_tmp.release()
    save_video_with_watermark(tmp_path, new_audio_path,
                              full_video_path, watermark=False)
    os.remove(tmp_path)
