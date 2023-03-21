# Model Evaluator

## Introduction

This is a simple tool to evaluate a few different models on a image. The models are:
- YOLOv7
- DETR
- PRBNet
- Detectron2 

## Installation

1. Clone the repository
2. `pip install -r requirements.txt`
3. `python detect.py`

Give the image path in the terminal when prompted relative to the detect.py file.

## Results

| Model                           | AP     | Inf. Time  |  
| ---                             |:------:| :---:      |
| YOLOv7-E6E                      |  56.8  |    18.7 ms |
| DETR                            |  42.0  |    36 ms   |
| PRBNet                          |  52.5  |    ~20.4ms |
| Detectron2 (Faster R-CNN FPN 3x)|  40.2  |   38 ms    | 


## Evaluation

In main folder, run `main.py` file. It will start the program. I do not recommend running the `detect.py` file because it contains experimental code.

## Testing Metrics

`metric_tester.py` file contains the code to test the metrics. It is not used in the main program. Detections and ground truth file path are given as arguments. The detections file should be in the below format:
"image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence" 

The ground truth file is the same as the detections file but without the confidence score. The output is the class based recall and precision, and the overall mAP@50, mAP@75, mAP score.

#### Note: Format of the models are changed to the above format. The original format for the models are not supported or obtainable anymore.

## References

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [DETR](https://github.com/facebookresearch/detr)
- [PRBNet](https://github.com/pingyang1117/PRBNet_PyTorch)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [MaskDINO](https://github.com/IDEA-Research/MaskDINO)