# This is a sample Python script.
import base64

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask, request, send_file, jsonify

from typing import List, Union
import Final2x_core as Fin
import os
import time
import cv2

app = Flask(__name__)

# 上传的图像文件将被保存在这个目录
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'


def upscale(picPATH: List[str]) -> None:
    config = Fin.SRCONFIG()
    config.inputpath = picPATH  # init log percentage
    config.tta = True
    config.model = 'RealESRGAN-anime'
    config.modelnoise = -1
    # ... see README.md for more config

    Fin.SR_queue()


def myupscale(filename):
    inputs = app.config['UPLOAD_FOLDER']
    outputs = app.config['OUTPUT_FOLDER']
    config = Fin.SRCONFIG()
    config.inputpath = [os.path.join(inputs,filename)]  # init log percentage
    config.tta = True
    config.model = 'RealESRGAN-anime'
    config.modelnoise = -1

    sr = Fin.SRFactory.getSR()
    # RGB Mode, RGBA can refer Final2x_core.SR_queue
    img = cv2.imread(os.path.join(inputs,filename), cv2.IMREAD_COLOR)
    img = sr.process(img)
    cv2.imwrite(os.path.join(outputs, filename), img)
    return filename


@app.route('/upload', methods=['POST'])
def upload_image():
    inputs = app.config['UPLOAD_FOLDER']
    outputs = app.config['OUTPUT_FOLDER']
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # 确保目录存在
        if not os.path.exists(inputs):
            os.makedirs(inputs)
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        # 生成时间戳
        timestamp = int(time.time())
        # 保存上传的文件
        filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join(inputs, filename)
        file.save(image_path)

        res_filename = myupscale(filename)

        # 现在您可以在img中访问图像数据，img_data中包含了图像的原始二进制数据

        # 在这里，您可以进行图像处理或其他操作
        # 返回处理后的图像
        # 读取图像数据并转为Base64编码
        with open(os.path.join(outputs,res_filename), 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')

        # 返回Base64编码的图像数据
        return jsonify({"image": img_data})
        # response = send_file(os.path.join(outputs, res_filename), mimetype='image/jpeg')

        # # 删除图像文件
        # os.remove(os.path.join(inputs, filename))
        # os.remove(os.path.join(outputs, res_filename))

        # return response


if __name__ == '__main__':
    app.run(debug=True)
