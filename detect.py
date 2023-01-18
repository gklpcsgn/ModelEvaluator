import sys
import os
import time

yoloPath = "./yolov7/yolov7"
detectorPath = "./detectron2/detectron2"
detrPath = "./detr/detr"

sys.path.append(yoloPath)
sys.path.append(detectorPath)
sys.path.append(detrPath)

# print(sys.path)

# Ask yolo or detectron2 to detect objects in the image

print("Please write the path to the image you want to detect objects in:")
imagePath = input()

print("Please write the model name you want to use [yolo/detectron2/detr]:")
modelName = input()

if modelName == "yolo":
    start_time = time.time()
    # os.system("conda activate yolov7")
    os.system("python3" + " " + yoloPath + "/detect.py" + " " + "--weights" + " " + yoloPath + "/yolov7-e6e.pt" + " " + "--conf" + " " + "0.60" + " " + "--img-size" + " " + "640" + " " + "--source" + " " + imagePath + " --save-txt")
    print("--- %s seconds ---" % (time.time() - start_time))
elif modelName == "detectron2":
    # os.system("conda activate detectron2")
    from Detector import *
    detector = Detector()
    start_time = time.time()
    detector.onImage(imagePath)
    print("--- %s seconds ---" % (time.time() - start_time))
elif modelName == "detr":
    # os.system("conda activate detr")
    from DETRDetector import *
    detector = DETRDetector()
    start_time = time.time()
    detector.onImage(imagePath)
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    print("Please write a valid model name")
