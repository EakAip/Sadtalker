# 接口：8002

# 数字人合成

from flask import Flask, request, jsonify, url_for
import os
import queue
import threading
from sadtalker_train import SadTalker
import uuid

app = Flask(__name__, static_folder='results')
sadtalker = SadTalker()  # 初始化
tasks = {}  # 存储任务状态和结果
tasks_lock = threading.Lock()  # 用于确保线程安全的锁
task_queue = queue.Queue()  # 创建一个队列用于管理任务

def process_video_task(avatarid, image_path, audio_path):
    try:
        infer_status, return_path = sadtalker.test(source_image=image_path, driven_audio=audio_path)
        with tasks_lock:
            if infer_status == 'processed':
                tasks[avatarid] = {"status": 2, 'remainder': 0, 'path': return_path}  # 更新为待生成URL状态
            else:
                tasks[avatarid] = {"status": 1, 'remainder': 180, 'url': ''}  # 更新为处理中状态
    except Exception as e:
        with tasks_lock:
            tasks[avatarid] = {"status": 5, 'remainder': 0, 'url': str(e)}  # 更新为异常状态

def worker():
    while True:
        avatarid, image_path, audio_path = task_queue.get()
        process_video_task(avatarid, image_path, audio_path)
        task_queue.task_done()

@app.route('/genvideo', methods=['POST'])
def generate_video():
    try:
        avatarid = request.form.get('avatarid')
        voicefile = request.files.get('voicefile')
        facefile = request.files.get('facefile')
        print(f"任务ID：{avatarid}")

        with tasks_lock:
            if avatarid in tasks:
                return jsonify({'code': 5, 'msg': 'Avatar ID already in use'}), 200

        if not facefile or not voicefile:
            return jsonify({'code': 5, 'msg': 'Missing source image or driven audio'}), 200

        uploads_dir = os.path.join(app.root_path, 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # 生成唯一的文件名
        image_filename = str(uuid.uuid4()) + os.path.splitext(facefile.filename)[1]
        audio_filename = str(uuid.uuid4()) + os.path.splitext(voicefile.filename)[1]
        
        image_path = os.path.join(uploads_dir, image_filename)
        audio_path = os.path.join(uploads_dir, audio_filename)
        
        facefile.save(image_path)
        voicefile.save(audio_path)

        tasks[avatarid] = {'status': 1, 'remainder': 180, 'url': ''}  # 初始化任务状态

        task_queue.put((avatarid, image_path, audio_path))

        return jsonify({
            'code': 0,
            'msg': 'OK',
            'data': {
                'avatarid': avatarid,
                'remainder': 180
            }
        }), 200
    except Exception as e:
        return jsonify({'code': 5, 'msg': str(e)}), 200


@app.route('/genstate', methods=['POST'])
def get_state():
    avatarid = request.form.get('avatarid')
    print(f"获取生产状态ID: {avatarid}")

    with tasks_lock:
        if avatarid not in tasks:
            return jsonify({"code": 5, "msg": "avatarid not in tasks"}), 200

        task = tasks[avatarid]

        # 如果状态是待生成URL，则在这里生成URL
        if task['status'] == 2:
            try:
                video_url = url_for('static', filename=task['path'][len('results/'):])

                full_url = request.url_root.rstrip('/') + "/digital_human" + video_url
                task['url'] = full_url
                task['status'] = 0  # 更新为成功状态
            except Exception as e:
                task['url'] = str(e)
                task['status'] = 5  # 更新为异常状态

    return jsonify({"code": task['status'],
                    "msg": "OK",
                    "data": {
                        "avatarid": avatarid,
                        "remainder": task['remainder'],
                        "url": task['url']
                    }}), 200

if __name__ == '__main__':
    threading.Thread(target=worker, daemon=True).start()
    app.run(port=8002, host='0.0.0.0')
