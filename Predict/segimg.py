from ultralytics import YOLO
from PIL import Image
import cv2
import config
from roboflow import Roboflow
import os
import numpy as np
import supervision as sv 
from typing import Union, List, Dict, Tuple

class SegResult:
    """分割检测结果的数据类"""
    def __init__(self, 
                boxes: np.ndarray,
                confidences: np.ndarray,
                cropped_images: List[np.ndarray],
                annotated: np.ndarray,
                saved_paths: List[str]):
        self.boxes = boxes
        self.confidences = confidences
        self.cropped_images = cropped_images
        self.annotated = annotated 
        self.saved_paths = saved_paths


def roboflow_predict(img: Union[str, np.ndarray, Image.Image], 
                    save_crops: bool = False,
                    output_dir: str = "./output_img") -> SegResult:
    """
    使用Roboflow API进行目标检测和分割。

    参数:
        img: Union[str, np.ndarray, Image.Image]
            输入图像，可以是:
            - 图像文件路径(str)
            - numpy数组(np.ndarray)
            - PIL图像对象(Image.Image)
        save_crops: bool, 默认False
            是否保存裁剪后的图像
        output_dir: str, 默认"./output_img"
            裁剪图像的保存目录

    返回:
        SegResult对象，包含:
            - boxes: np.ndarray, 检测框坐标 [x1,y1,x2,y2]
            - confidences: np.ndarray, 置信度scores
            - cropped_images: List[np.ndarray], 裁剪后的图像列表
            - annotated: np.ndarray, 标注的原始图像
            - saved_paths: List[str], 保存的图像路径列表
    """
    # Roboflow设置
    api_key = config.ROBOFLOW_API_KEY
    if not api_key:
        raise ValueError("请在config文件中设置ROBOFLOW_API_KEY")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("dermnet-segmentation")
    model = project.version(2).model

    # 预测
    result = model.predict(img, confidence=config.SEG_MODEL_CONFIDENCE).json()
    predictions = result["predictions"]

    # 提取预测结果
    boxes = []
    confidences = []
    class_ids = []
    cropped_images = []
    saved_paths = []

    # 读取图像
    if isinstance(img, str):
        image = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        image = img.copy()
    elif isinstance(img, Image.Image):
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("不支持的图像格式")

    # 创建标注器
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 生成标注
    labels = [pred["class"] for pred in predictions]
    detections = sv.Detections(
        xyxy=np.array([[pred["x"] - pred["width"] / 2,
                        pred["y"] - pred["height"] / 2,
                        pred["x"] + pred["width"] / 2,
                        pred["y"] + pred["height"] / 2] for pred in predictions]),
        confidence=np.array([pred["confidence"] for pred in predictions]),
        class_id=np.array([0 for _ in predictions])
    )

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    for pred in predictions:
        # 提取边界框坐标
        x1 = pred["x"] - pred["width"] / 2
        y1 = pred["y"] - pred["height"] / 2
        x2 = x1 + pred["width"]
        y2 = y1 + pred["height"]
        boxes.append([x1, y1, x2, y2])
        confidences.append(pred["confidence"])
        class_ids.append(0)

        # 扩展边界并裁剪
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(image.shape[1], x2 + 5)
        y2 = min(image.shape[0], y2 + 5)
        
        cropped = image[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, ((x2 - x1) * 2, (y2 - y1) * 2))
        cropped_images.append(cropped)
            
        if save_crops:
            # 保存图片
            os.makedirs(output_dir, exist_ok=True)
            segimg_path = os.path.join(output_dir, f'cropped_{len(boxes)-1}.jpg')
            cv2.imwrite(segimg_path, cropped)
            saved_paths.append(segimg_path)
    if save_crops:
        # 保存标注的原始图像
        annotated_path = os.path.join(output_dir, 'annotated.jpg')
        cv2.imwrite(annotated_path, annotated)
        saved_paths.append(annotated_path)



    return SegResult(
        boxes=np.array(boxes),
        confidences=np.array(confidences),
        cropped_images=cropped_images,
        annotated=annotated,
        saved_paths=saved_paths
    )


def seg_img(img: Union[str, np.ndarray, Image.Image],
            save_crops: bool = False,
            output_dir: str = "./output_img") -> SegResult:
    """
    使用本地YOLO模型进行目标检测和分割，如果失败则回退到Roboflow API。

    参数:
        img: Union[str, np.ndarray, Image.Image]
            输入图像，可以是:
            - 图像文件路径(str)
            - numpy数组(np.ndarray)
            - PIL图像对象(Image.Image)
        save_crops: bool, 默认True
            是否保存裁剪后的图像和标注的原图
        output_dir: str, 默认"./output_img"
            裁剪图像的保存目录

    返回:
        SegResult对象，包含:
            - boxes: np.ndarray, 检测框坐标 [x1,y1,x2,y2]
            - confidences: np.ndarray, 置信度scores
            - cropped_images: List[np.ndarray], 裁剪后的图像列表
            - annotated: np.ndarray, 标注的原始图像
            - saved_paths: List[str], 保存的图像路径列表
    """
    model = YOLO(config.SEG_MODEL_PATH)
    
    try:
        # 读取图像
        if isinstance(img, str):
            image = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            image = img.copy()
        elif isinstance(img, Image.Image):
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("不支持的图像格式")

        results = model.predict(img, 
                                config.YOLO_LINE_WIDTH, 
                                conf=config.YOLO_CONFIDENCE)
        # 转换为列表
        results = list(results)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("本地模型未检测到目标，尝试使用API进行检测")
            return roboflow_predict(img, save_crops, output_dir)

        # 生成标注的原始图像
        annotated = results[0].plot()
        
        boxes_list = []
        confidences = []
        cropped_images = []
        saved_paths = []

        for idx, box in enumerate(boxes):
            confidence = box.conf.item()
            box_xyxy = box.xyxy.tolist()[0]
            boxes_list.append(box_xyxy)
            confidences.append(confidence)

            x1, y1, x2, y2 = map(int, box_xyxy)
            # 扩展边界
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(image.shape[1], x2 + 5)
            y2 = min(image.shape[0], y2 + 5)
            
            # 图片扩大两倍
            cropped = cv2.resize(image[y1:y2, x1:x2], ((x2 - x1) * 2, (y2 - y1) * 2))
            cropped_images.append(cropped)
                
            if save_crops:
                # 保存图片
                os.makedirs(output_dir, exist_ok=True)
                segimg_path = os.path.join(output_dir, f'cropped_{idx}.jpg')
                cv2.imwrite(segimg_path, cropped)
                saved_paths.append(segimg_path)

        if save_crops:
            # 保存标注的原始图像
            annotated_path = os.path.join(output_dir, 'annotated.jpg')
            cv2.imwrite(annotated_path, annotated)
            saved_paths.append(annotated_path)

        return SegResult(
            boxes=np.array(boxes_list),
            confidences=np.array(confidences),
            annotated=annotated,
            cropped_images=cropped_images,
            saved_paths=saved_paths
        )

    except Exception as e:
        print(f"本地模型检测出错：{e}，使用API进行检测")
        return roboflow_predict(img, save_crops, output_dir)
    
# if __name__ == "__main__":
#     # 测试
#     response = seg_img("./BCC基底细胞癌.jpg")

#     boxes = response.boxes  # 获取检测框坐标, 返回两张切割图片坐标的示例: [[x1,y1,x2,y2], [x1,y1,x2,y2]]
#     confidences = response.confidences  # 获取置信度, 返回示例: [0.9, 0.8, 0.7]
#     cropped_images = response.cropped_images  # 获取裁剪后的图像,为numpy数组列表
#     annotated = response.annotated  # 获取标注的原始图像,为numpy数组
#     saved_paths = response.saved_paths  # 获取保存的图像路径, 为字符串列表