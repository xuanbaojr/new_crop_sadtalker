import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid
from rembg import remove

from src.utils.videoio import save_video_with_watermark


def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):

    full_img = cv2.imread(pic_path)
    if not os.path.isfile(pic_path):
        print("pic_path is not exist!")
    if not os.path.isfile(video_path):
        print("video_path is not exist!")

    w, h = full_img.shape[:2]
    full_img = cv2.resize(full_img, (256, 256))
    cv2.imwrite('./pic_path.png', full_img)
    pic_path = './pic_path.png'

    output_path = './test/test4.png'

    with open(pic_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)
    test4 = cv2.imread(output_path)

    test4_temp = cv2.resize(test4, (256, 256))

    blur_img = cv2.cvtColor(test4, cv2.COLOR_BGR2GRAY)
    adaptive_threshold_image = cv2.adaptiveThreshold(
        blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 2)

    cv2.imwrite('test/threshold_img.png', adaptive_threshold_image)
    kernel = np.ones((2, 1), np.uint8)
    adaptive_threshold_image = cv2.dilate(
        adaptive_threshold_image, kernel, iterations=11)

    cv2.imwrite('test/threshold_img_kernel.png', adaptive_threshold_image)

    contour_img = adaptive_threshold_image.copy()
    contours, _ = cv2.findContours(
        contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros((h+2, w+2), dtype=np.uint8)
    cv2.drawContours(test4, [largest_contour], 0, (255, 0, 0), thickness=1)
    cv2.imwrite('./test/test4_test.png', test4)
    cv2.imwrite('test/test4_draw.png', test4)

    target_color = [255, 0, 0]
    indices = np.where(np.all(test4 == target_color, axis=-1))
    points = list(zip(indices[0], indices[1]))

    # comprehension
    array_y = []
    for x in range(w):
        col_indices = [point for point in points if point[1] == x]
        if col_indices:
            array_y.append(col_indices[0])

    array_x = []
    for y in range(h):
        row_indices = [point for point in points if point[0] == y]
        if row_indices:
            array_x.extend([row_indices[0], row_indices[-1]])

    unique_points = set(array_x + array_y)
    print(array_y[0], array_y[-1])

    yl_ = min(5, array_y[0][1])
    yr = min(10, (w - array_y[-1][1]-1))

    # for x, y in unique_points:
    #     if y < w/2:
    #         yl = min(5, y)
    #         full_img[x,y:y+yl] = np.copy(full_img[x,y-yl:y])
    #     if y > w/2:
    #         full_img[x,y-yr:y+1] = np.copy(full_img[x, y:y+yr+1])

    #   unique_points_sorted = sorted(unique_points, key=lambda point: (point[0], point[1]))

    cv2.imwrite('test/test4_draw_array.png', test4)
    w = full_img.shape[0]
    h = full_img.shape[1]
    full_img = full_img.astype(np.uint8)
    cv2.imwrite('./test/full_img.png', full_img)

    test3 = cv2.imread('./test/full_img.png')
    test3 = cv2.resize(test3, (256, 256))
    test3_blur = cv2.blur(test3, (51, 51))
    test3_ = np.ones_like(test3)*120

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)

    print("crop_frame", crop_frames[0].shape)
    cv2.imwrite('./test/haha.png', crop_frames[0])

    tmp_path = './test/output.mp4'
    out_tmp = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    for frame in tqdm(crop_frames, 'SeamlessClone:'):
        test4 = cv2.resize(frame, (256, 256))
        test4_temp = cv2.resize(test4, (256, 256))

        blur_img = cv2.cvtColor(test4, cv2.COLOR_BGR2GRAY)
        adaptive_threshold_image = cv2.adaptiveThreshold(
            blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 2)

        cv2.imwrite('test/threshold_img.png', adaptive_threshold_image)

        kernel = np.ones((2, 1), np.uint8)
        adaptive_threshold_image = cv2.dilate(
            adaptive_threshold_image, kernel, iterations=11)
        cv2.imwrite('test/threshold_img_kernel.png', adaptive_threshold_image)

        contour_img = adaptive_threshold_image.copy()
        contours, _ = cv2.findContours(
            contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros((h+2, w+2), dtype=np.uint8)
        cv2.drawContours(test4, [largest_contour], 0, (255, 0, 0), thickness=1)
        cv2.imwrite('./test/test4_test.png', test4)
        cv2.imwrite('test/test4_draw.png', test4)

        if (test4[0, 0] == [255, 0, 0]).all():

            break

        target_color = [255, 0, 0]

        indices = np.where(np.all(test4 == target_color, axis=-1))

        points = list(zip(indices[0], indices[1]))

        # comprehension
        array_y = []
        for x in range(w):
            col_indices = [point for point in points if point[1] == x]
            if col_indices:
                array_y.append(col_indices[0])

        array_x = []
        for y in range(h):
            row_indices = [point for point in points if point[0] == y]
            if row_indices:
                array_x.extend([row_indices[0], row_indices[-1]])

        unique_points = set(array_x + array_y)

        for y, x in unique_points:
            test4[y, x] = [100, 100, 255]

        cv2.imwrite('test/test4_draw_array.png', test4)

        loDiff = (45, 45, 45)
        upDiff = (80, 80, 80)
        cv2.floodFill(test4, mask, (0, 0), (255, 255, 255), loDiff, upDiff)
        cv2.floodFill(test4, mask, (w-1, 0), (255, 255, 255), loDiff, upDiff)

        cv2.imwrite('test/test4_fill.png', test4)

        blur_img = cv2.cvtColor(test4, cv2.COLOR_BGR2GRAY)

        test4_temp[np.where(np.all(test4 == [255, 0, 0], axis=2))] = np.copy(
            test4_temp[np.where(np.all(test4 == [255, 0, 0], axis=2))])
        test4_temp[np.where(np.all(test4 == [255, 255, 255], axis=2))] = np.copy(
            test3[np.where(np.all(test4 == [255, 255, 255], axis=2))])
        test4_temp[np.where(np.all(test4 == [100, 100, 255], axis=2))] = np.copy(
            test3_blur[np.where(np.all(test4 == [100, 100, 255], axis=2))])

        cv2.imwrite('test/contour.png', test3)
        cv2.imwrite('test/test4_.png', test4_temp)
        out_tmp.write(test4_temp)
    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path,
                              full_video_path, watermark=False)
