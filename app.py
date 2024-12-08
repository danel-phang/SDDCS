from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import config
from Predict.segimg import seg_img, roboflow_predict
from typing import Union, List, Dict, Tuple
import gradio as gr
import io

class DetectionResult:
    """检测结果的数据类"""
    def __init__(self, detected: bool, image: Image, boxes: dict, orig_shape: tuple, speed: dict):
        self.detected = detected
        self.image = image
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.speed = speed

def predict_img(img) -> Image.Image:
    """
    参数:
    :param img: numpy array or PIL Image
    :return: PIL Image对象
    """
    model = YOLO(config.YOLO_MODEL_PATH)
    results = list(model.predict(img, config.YOLO_LINE_WIDTH, conf=config.YOLO_CONFIDENCE))
    
    if results[0].boxes.shape[0] > 0:
        result_array = results[0].plot()
        result_array_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_array_rgb)
    return None

def multi_predict(img: Union[str, np.ndarray, Image.Image]) -> Tuple[Image.Image, List[Image.Image]]:
    """
    参数:
    :param img: 输入图像
    :return: Tuple(标注后的图像, 预测图像列表)
    """
    response = seg_img(img)
    
    # 获取检测结果
    cropped_images = response.cropped_images
    annotated = response.annotated
    
    # 处理预测图像
    predict_images = []
    for cropped_img in cropped_images:
        predicted_img = predict_img(cropped_img)
        if predicted_img:
            predict_images.append(predicted_img)
    
    # 转换标注图像
    annotated_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    
    return annotated_image, predict_images

def process_image(input_image) -> Tuple[Image.Image, List[Image.Image]]:
    """
    参数:
    :param input_image: 输入图像
    :return: Tuple(标注图像, 预测图像列表)
    """
    # 统一输入图像格式为BGR的numpy数组
    if isinstance(input_image, Image.Image):
        input_array = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    elif isinstance(input_image, np.ndarray):
        input_array = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    elif isinstance(input_image, str):
        input_array = cv2.imread(input_image)
    else:
        raise ValueError("Unsupported image format")
    
    return multi_predict(input_array)

with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>图像检测与预测</h1>")
    with gr.Row():
        input_image = gr.Image(label="上传图像")
    with gr.Row():
        annotated_button = gr.Button("提交")
    with gr.Row():
        annotated_output = gr.Image(label="整体识别")
    with gr.Row():
        predict_outputs = gr.Gallery(label="预测图像")

    annotated_button.click(process_image, inputs=input_image, outputs=[annotated_output, predict_outputs])

demo.launch(server_port=6626, share=True)