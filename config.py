# config.py

# YOLO模型路径
YOLO_MODEL_PATH = r"./models/predict.pt"
#预测置信度阈值,高于该值 detected 返回True,反之则反 (范围0-1)
YOLO_CONFIDENCE = 0.2
#预测框文字大小
YOLO_LINE_WIDTH = 1

# 分割模型路径
SEG_MODEL_PATH = r"./models/segementation180.pt"
# 分割置信度阈值 (范围0-100)
SEG_MODEL_CONFIDENCE = 20

# Roboflow API密钥
ROBOFLOW_API_KEY = "18oZy6U3mMau0dnIXs66"