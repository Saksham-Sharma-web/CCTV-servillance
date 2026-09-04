# Model Weights Directory

This directory is the designated local storage location for model weights:

1. `yolov8n.pt` (Object Detection, automatically downloaded by Ultralytics if omitted)
2. `face_detection_yunet_2023mar.onnx` (YuNet Face Detection, optional; OpenCV Haar cascade used as universal fallback if omitted)

Callers can specify custom model paths via `IBVAPConfig(models_dir="...")` or by passing custom model paths directly to `IBVAPPipeline`.
