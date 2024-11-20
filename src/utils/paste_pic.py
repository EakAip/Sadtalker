
# 模型加速 充分利用cpu性能

# 多进程处理

# 内存垃圾回收处理（其实没必要，后面步骤会做处理）


import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid
import psutil
import gc
from multiprocessing import Pool, cpu_count
from src.utils.videoio import save_video_with_watermark

def process_frame(crop_frame, full_img, ox1, ox2, oy1, oy2):
    """
    处理单个视频帧，将其无缝合成到目标图像
    """
    p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1)) 

    mask = 255 * np.ones(p.shape, p.dtype)
    location = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
    gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)

    return gen_img

def monitor_cpu_usage_and_adjust_pool():
    """
    监控当前CPU使用率，并根据其动态调整进程池的大小
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"当前CPU使用率: {cpu_usage}%")

    # 如果CPU使用率高于80%，减少并行任务的数量
    if cpu_usage > 80:
        return max(cpu_count() // 2, 1)  # 使用一半的CPU核心
    elif cpu_usage < 50:
        return cpu_count()  # 使用所有的CPU核心
    else:
        return cpu_count() // 2  # 使用一半的CPU核心

def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):
    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        # loader for first frame
        full_img = cv2.imread(pic_path)
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = [] 
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break 
            break 
        full_img = frame

    frame_h = full_img.shape[0]
    frame_w = full_img.shape[1]

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)

    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    tmp_path = str(uuid.uuid4()) + '.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))

    # 创建进程池，并行处理视频帧
    with Pool(cpu_count()) as pool:
        print(f"开始seamlessClone,使用的CPU核心数：{cpu_count()}")
        results = []
        for crop_frame in tqdm(crop_frames, 'seamlessClone:'):
            
            # 将每帧处理任务传递给进程池
            results.append(pool.apply_async(process_frame, (crop_frame, full_img, ox1, ox2, oy1, oy2)))

        # 等待所有进程完成并收集结果
        for result in results:
            gen_img = result.get()
            out_tmp.write(gen_img)
        
        # # 垃圾清理
        # print("开始垃圾清理")
        # gc.collect()


    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)

