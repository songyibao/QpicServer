# This is a sample Python script.
import base64
import tensorflow as tf
import torch

from flask import Flask, request
from typing import List
import Final2x_core as Fin
import os
import time
import cv2
from PIL import Image

from flask_restful import Resource, Api
from style_transfer import transfer

app = Flask(__name__)

# 上传的图像文件将被保存在这个目录
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['STYLE_UPLOAD_FOLDER'] = 'style_transfer/images'


def upscale(picPATH: List[str]) -> None:
    config = Fin.SRCONFIG()
    config.gpuid = 0
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
    config.gpuid = 0
    config.inputpath = [os.path.join(inputs, filename)]  # init log percentage
    config.tta = True
    config.model = 'RealESRGAN-anime'
    config.modelnoise = -1

    sr = Fin.SRFactory.getSR()
    # RGB Mode, RGBA can refer Final2x_core.SR_queue
    img = cv2.imread(os.path.join(inputs, filename), cv2.IMREAD_COLOR)
    img = sr.process(img)
    cv2.imwrite(os.path.join(outputs, filename), img)
    return filename


def pic2anime(filename):
    inputs = app.config['UPLOAD_FOLDER']
    outputs = app.config['OUTPUT_FOLDER']
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
    input_image = os.path.join(inputs, filename)
    img = Image.open(input_image).convert("RGB")
    out = face2paint(model, img)
    out.save(os.path.join(outputs, filename))
    torch.cuda.empty_cache()
    return filename


def my_style_transfer(content_image_path, style_image_path, filename_only):
    TS = transfer.NeuralStyleTransfer(content_image_path, style_image_path)
    TS.start(filename_only)
    tf.keras.backend.clear_session()
    return filename_only + '.jpg'


# @app.route('/upload', methods=['POST'])
def upload_image(file, type):
    inputs = app.config['UPLOAD_FOLDER']
    outputs = app.config['OUTPUT_FOLDER']
    if 'file' not in request.files:
        return "No file part"

    # file = request.files['file']

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
        # filename = f"{timestamp}_{file.filename}"
        ext_name = os.path.splitext(file.filename)[-1]
        ext_name = ext_name.lower()
        filename = f"{timestamp}{ext_name}"
        # print(filename)
        image_path = os.path.join(inputs, filename)
        file.save(image_path)
        res_filename = ""
        if type == 1:
            res_filename = myupscale(filename)
        elif type == 2:
            res_filename = pic2anime(filename)
        # 现在您可以在img中访问图像数据，img_data中包含了图像的原始二进制数据

        # 在这里，您可以进行图像处理或其他操作
        # 返回处理后的图像
        # 读取图像数据并转为Base64编码
        with open(os.path.join(outputs, res_filename), 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        img = Image.open(os.path.join(outputs, res_filename))
        ext_name = img.format.lower()
        del img
        # 返回Base64编码的图像数据
        # return "data:image/" + ext_name + ";base64," + img_data
        response = "data:image/" + ext_name + ";base64," + img_data

        # 删除图像文件
        os.remove(image_path)
        os.remove(os.path.join(outputs, res_filename))

        return response


def upload_two_images(files):
    inputs = app.config['UPLOAD_FOLDER']
    outputs = app.config['OUTPUT_FOLDER']
    # if 'file' not in request.files:
    #     return "No file part"
    print(files)
    if len(files) != 2:
        return "Error files count"
    # file = request.files['file']

    if files[0].filename == '' or files[1].filename == '':
        return "No selected file"

    if files:
        # 确保目录存在
        if not os.path.exists(inputs):
            os.makedirs(inputs)
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        # 生成时间戳
        timestamp = int(time.time())
        content_filename = f"{timestamp}_{files[0].filename}"
        style_filename = f"{timestamp}_{files[1].filename}"
        content_image_path = os.path.join(inputs, content_filename)
        files[0].save(content_image_path)
        style_image_path = os.path.join(inputs, style_filename)
        files[1].save(style_image_path)
        res_filename = my_style_transfer(content_image_path, style_image_path, str(timestamp))
        # 返回处理后的图像
        # 读取图像数据并转为Base64编码
        with open(os.path.join(outputs, res_filename), 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        img = Image.open(os.path.join(outputs, res_filename))
        ext_name = img.format.lower()
        del img
        del files
        # 返回Base64编码的图像数据
        # return "data:image/" + ext_name + ";base64," + img_data
        response = "data:image/" + ext_name + ";base64," + img_data

        # 删除图像文件
        os.remove(content_image_path)
        os.remove(style_image_path)
        os.remove(os.path.join(outputs, res_filename))

        return response


class UploadImg(Resource):

    # def __init__(self):
    #     # 创建一个新的解析器
    #     self.parser = reqparse.RequestParser()
    #     # 增加imgFile参数，用来解析前端传来的图片。
    #     self.parser.add_argument(
    #         'imgFile',
    #         required=True,
    #         type=FileStorage,
    #         location='files',
    #         help="imgFile is wrong.")

    def post(self, id):
        img_file = request.files['file']
        image_base64_str = upload_image(img_file, id)
        return {'data': image_base64_str}


class UploadImgs(Resource):


    def post(self):
        img_files = request.files
        print(img_files)
        img_arr = img_files.getlist('file0')
        img_arr = img_arr + img_files.getlist('file1')
        image_base64_str = upload_two_images(img_arr)
        return {'data': image_base64_str}


if __name__ == '__main__':
    api = Api(app)
    # id 1->超分辨率 2->图像转动漫 3->图像风格迁移
    api.add_resource(UploadImg, '/uploadImg/<int:id>')
    api.add_resource(UploadImgs, '/uploadImg/styleTransfer')
    app.run(debug=True, host="0.0.0.0", port=5000)
