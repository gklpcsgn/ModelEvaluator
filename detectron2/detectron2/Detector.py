from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import cv2,numpy as np
import json
import os 
from pathlib import Path


try:
    with open('../catagories.json') as f:
        categories = json.load(f)
except:
    print("Error: categories.json not found")

class Detector:
    def __init__(self,confidence_threshold=0.7):
        self.cfg = get_cfg()

        # Load model and weights
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # set threshold for this model
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self,imagePath,save_img=False,save_img_path=None,model_name=None):
        image = cv2.imread(imagePath)
        outputs = self.predictor(image)
        boxes = outputs["instances"].pred_boxes
        classes = outputs["instances"].pred_classes
       

        v = Visualizer(image[:, :, ::-1],metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),instance_mode=ColorMode.IMAGE_BW)
        
        
        # print(str(boxes))
        # print(str(classes))

        # print(outputs["instances"].to("cpu").scores)

        # get names of classes
        # ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)
        # print(class_names)

        # if there is no folder named detectron2_output, create one
        if not os.path.exists("detectron2_output"):
            os.makedirs("detectron2_output")
        # we will have a folder for each image in the detectron2_output folder
        # if there is no folder named detectron2_output/image_name, create one. If there is, add _1, _2, _3, etc.
        if not os.path.exists("detectron2_output/"+imagePath.split("/")[-1].split(".")[0]):
            save_path = "detectron2_output/"+imagePath.split("/")[-1].split(".")[0]
            os.makedirs(save_path)
        else:
            i = 1
            while os.path.exists("detectron2_output/"+imagePath.split("/")[-1].split(".")[0]+"_"+str(i)):
                i += 1

            save_path = "detectron2_output/"+imagePath.split("/")[-1].split(".")[0]+"_"+str(i)
            os.makedirs(save_path)

        label_id = 0
        temp = []
        coordinates = boxes.tensor.cpu().numpy()
        # get 1 decimal place
        coordinates = np.around(coordinates, decimals=1)

        # print(coordinates.shape)
        for i in range(coordinates.shape[0]):
            # append name of the image, label_id, x1, y1, x2, y2, xcenter,  ycenter, label, confidence
            temp.append(imagePath.split("/")[-1].split(".")[0])
            temp.append(label_id)
            temp.append(coordinates[i][0])
            temp.append(coordinates[i][1])
            temp.append(coordinates[i][2])
            temp.append(coordinates[i][3])
            temp.append(np.around((coordinates[i][0]+coordinates[i][2])/2, decimals=1))
            temp.append(np.around((coordinates[i][1]+coordinates[i][3])/2, decimals=1))

            name = str(class_names[classes[i]]).lower()
            name_id = "nan"
            for j in range(len(categories)):
                if categories[j]['name'] == name:
                    name_id = categories[j]['id']
                    break
            if name_id == "nan":
                print("Error: name not found")
                break
            # print(name_id)
            # print(name)
            temp.append(name_id)

            temp.append(np.around(outputs["instances"].to("cpu").scores[i].item(),decimals=4))

            label_id += 1
            # convert to string
            temp = [str(x) for x in temp]
            # join list items by comma
            temp = ','.join(temp)

            # print(temp)

            if save_img_path is None:
                with open(save_path+"/"+"labels.txt", "a") as f:
                    f.write(temp)
                    f.write("\n")
                f.close()
            else:
                with open(save_img_path, "a") as f:
                    f.write(temp)
                    f.write("\n")
                f.close()
            temp = []  # clear list



        
        # if save_img:
        #     print("Saving image...")
        #     print(save_img_path)
        #     out = v.draw_instance_predictions_default(outputs["instances"].to("cpu"))
        #     cv2.imwrite(save_img_path,out.get_image()[:, :, ::-1])
        
        # cv2.imshow("image",out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
