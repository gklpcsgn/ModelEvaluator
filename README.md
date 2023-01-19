# Model Evaluator

### Introduction

This is a simple tool to evaluate a few different models on a image. The models are:
- YOLOv7
- DETR
- PRBNet
- Detectron2 

### Installation

1. Clone the repository
2. `pip install -r requirements.txt`
3. `python detect.py`

Image path cannot be given as an argument yet. It will be added soon. Give the image path in the terminal when prompted relative to the detect.py file.

### Results

| Model                           | AP     | Inf. Time             | 
| ---                             |:------:| :---:                 |
| YOLOv7-E6E                      |  56.8  |    18.7 ms  |
| DETR                            |  42.0  |    36 ms              |
| PRBNet                          |  52.5  |    ~20.4ms            |
| Detectron2 (Faster R-CNN FPN 3x)|  40.2  |   38 ms               | 
