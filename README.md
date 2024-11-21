# 数字人合成

1.新增模型修改为加速版本

2.新增显存回收，垃圾清理

后期工作：

1.模型全部改为绝对路径

2.增加程序保活机制

## 创建conda环境
```python
conda create -n sadtalker python=3.8
conda activate sadtalker
```

## 安装依赖包和ffmpeg
```python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c conda-forge ffmpeg
pip install tb-nightly -i https://pypi.org/simple # 这个包需要使用国外源安装
pip install -r requirements.txt
```

## 下载模型
```python
bash scripts/download_models.sh 
```
国内下载模型很慢 百度网盘链接：https://pan.baidu.com/s/1kbF656qtJCrhTXfTusZ12g?pwd=0000 

## 运行
```python
nohup python sadtalker_flask.py & tail -f nohup.out
```

