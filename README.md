# 数字人合成

## 创建conda环境
```python
conda create -n sadtalker python=3.8
conda activate sadtalker
```

## 安装依赖包和ffmpeg
```python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install ffmpeg
pip install tb-nightly -i https://pypi.org/simple # 这个包需要使用国外源安装
pip install -r requirements.txt
```

## 下载模型
```python
bash scripts/download_models.sh
```

## 运行
```python
nohup python sadtalker_flask.py & tail -f nohup.out
```
