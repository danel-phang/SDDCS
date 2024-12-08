# 皮肤疾病检测与分类系统

## 项目介绍

本项目是一个基于深度学习的皮肤疾病检测与分类系统。系统使用 YOLO 模型进行皮肤病变区域的检测和分割，并对分割后的区域进行疾病分类。该系统可以帮助用户快速识别潜在的皮肤健康问题。

## 系统要求

- Python 3.10+
- CUDA 支持（推荐）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone git@github.com:danel-phang/SDDCS.git
cd 1-Projiect
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
   - 将检测模型 `predict.pt` 放入 `./models/` 目录
   - 将分割模型 `segementation180.pt` 放入 `./models/` 目录

## 配置说明

在 `config.py` 文件中，你可以修改以下配置：

- `YOLO_CONFIDENCE`: 检测置信度阈值（默认0.1）
- `YOLO_LINE_WIDTH`: 预测框线宽（默认1）
- `SEG_MODEL_CONFIDENCE`: 分割模型置信度阈值（默认1）
- `ROBOFLOW_API_KEY`: Roboflow API密钥（用于备用检测）

## 使用方法

1. 启动：
```bash
python app.py
```

2. 访问Web界面：
   - 打开浏览器，访问 `http://localhost:6626`
   - 或使用命令行显示的公共URL访问

3. 使用步骤：
   - 点击"上传图像"按钮选择需要检测的皮肤图片
   - 点击"提交"按钮开始检测
   - 系统将显示两类结果：
     - 整体识别：显示原图中所有检测到的病变区域
     - 预测图像：显示每个分割区域的具体分类结果

## 结果说明

- **整体识别**：
  - 在原图上标注所有检测到的疑似病变区域
  - 每个标注框包含位置信息和置信度

- **预测图像**：
  - 显示每个分割区域的放大图像
  - 包含具体的疾病分类结果
  - 置信度分数显示在标注中

## 注意事项

1. 图片要求：
   - 支持常见图片格式（jpg、png、jpeg等）
   - 建议使用清晰、光线充足的图片
   - 图片大小建议不超过4MB

2. 检测限制：
   - 单次检测支持一张图片
   - 处理时间与图片大小和复杂度相关

3. 使用建议：
   - 拍摄时保持光线充足
   - 对准病变部位，避免模糊
   - 必要时使用多个角度拍摄

## 常见问题

1. Q: 系统无法启动？
   A: 检查Python环境和依赖安装是否完整，确保模型文件位置正确。

2. Q: 检测结果不准确？
   A: 可以尝试：
   - 调整图片光线和清晰度
   - 在config.py中调整置信度阈值
   - 尝试不同角度拍摄

3. Q: 系统反应较慢？
   A: 检查是否有GPU支持，可以适当调整图片大小。

## 项目结构

```
1-Projiect/
├── models/               # 模型文件目录
├── Predict/             # 预测相关代码
├── output_img/          # 输出图像保存目录
├── app.py               # Web应用主程序
├── config.py            # 配置文件
├── requirements.txt     # 依赖清单
└── readme.md            # 说明文档
```

## 免责声明

本系统仅供参考，不能替代专业医生的诊断。如有皮肤健康问题，请及时就医。

## 技术支持

如有技术问题，请提交issue或联系技术支持。
