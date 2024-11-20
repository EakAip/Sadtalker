# flask接口使用，勿删

# 去除enhance 

# 对低质量视频增加处理模块

# 模型加速处理

# 增加显存回收模块



import os
import gc
import shutil
import glob
import subprocess
import torch, uuid
import os, sys, shutil
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff     # 模型加速修改
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from pydub import AudioSegment
from flask import url_for
from flask import request
import logging

logging.basicConfig(level=logging.ERROR)  # 设置日志级别为 ERROR，避免显示 INFO 或 DEBUG 级别的信息


# 将MP3格式的音频文件转换为WAV格式
def mp3_to_wav(mp3_filename,wav_filename,frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename,format="wav")
    

def convert_video_for_browser(input_path, output_path):
    # 使用 ffmpeg 将视频转码为 H.264 和 AAC 音频编码的 MP4 格式
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",          # 视频编码格式设置为 H.264
        "-profile:v", "high",        # 使用 High Profile，以提高兼容性和质量
        "-c:a", "aac",               # 音频编码设置为 AAC
        "-b:a", "128k",              # 设置音频比特率
        "-movflags", "faststart",    # 添加 faststart 标志，优化浏览器播放
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"视频已成功转换为浏览器兼容格式，保存路径为: {output_path}")
    except subprocess.CalledProcessError as e:
        print("视频转换失败:", e)

class SadTalker():
    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        self.device = device
        os.environ['TORCH_HOME'] = checkpoint_path
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path

    def clear_memory(self):
        """
        Clear GPU memory and perform garbage collection
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空缓存
            torch.cuda.synchronize()  # 等待所有CUDA操作完成，确保显存释放
        gc.collect()  # 调用Python的垃圾回收机制

    def test(self, source_image, driven_audio, preprocess='full', 
             still_mode=True, use_enhancer=False, batch_size=24, size=256, 
             pose_style=0, exp_scale=1.0, use_ref_video=False, ref_video=None,
             ref_info=None, use_idle_mode=False, length_of_audio=0, use_blink=True,
             result_dir='results'):

        self.sadtalker_paths = init_path(self.checkpoint_path, self.config_path, size, False, preprocess)
        
        self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
        self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(self.sadtalker_paths, self.device)

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.move(source_image, input_dir)

        # 处理音频
        if driven_audio is not None and os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  
            if '.mp3' in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace('.mp3', '.wav'), 16000)
                audio_path = audio_path.replace('.mp3', '.wav')
            else:
                shutil.move(driven_audio, input_dir)

        elif use_idle_mode:
            audio_path = os.path.join(input_dir, 'idlemode_'+str(length_of_audio)+'.wav')  
            from pydub import AudioSegment
            one_sec_segment = AudioSegment.silent(duration=1000*length_of_audio)
            one_sec_segment.export(audio_path, format="wav")
        
        os.makedirs(save_dir, exist_ok=True)

        # 处理输入图片，生成3DMM系数
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(pic_path, first_frame_dir, preprocess, True, size)
        
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        if use_ref_video:
            ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
            ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
            os.makedirs(ref_video_frame_dir, exist_ok=True)
            ref_video_coeff_path, _, _ = self.preprocess_model.generate(ref_video, ref_video_frame_dir, preprocess, source_image_flag=False)
        else:
            ref_video_coeff_path = None

        # 音频转换为系数
        batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=None, still=still_mode, idlemode=use_idle_mode, length_of_audio=length_of_audio, use_blink=use_blink)
        coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path=None)

        # 转换系数为视频
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode, preprocess=preprocess, size=size, expression_scale=exp_scale)
        return_path = self.animate_from_coeff.generate(data, save_dir, pic_path, crop_info, enhancer='gfpgan' if use_enhancer else None, preprocess=preprocess, img_size=size)
        video_name = data['video_name']

        print(f'The generated video is named {video_name} in {save_dir}')
        
        # 处理视频以适应浏览器播放
        result_path = os.path.join(os.path.dirname(return_path), f"{os.path.splitext(os.path.basename(return_path))[0]}_convert.mp4")
        convert_video_for_browser(return_path, result_path)

        # 清理模型和显存
        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff
        
        # 执行显存回收和垃圾回收
        self.clear_memory()

        return 'processed', result_path
