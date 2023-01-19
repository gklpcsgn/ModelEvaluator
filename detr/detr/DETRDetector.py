import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
import os
from PIL import Image
import numpy as np
import json

try:
    with open('../catagories.json') as f:
        categories = json.load(f)
except:
    print("Error: categories.json not found")

# COCO classes
class_names = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes,imagePath,save_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        # print (c)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{class_names[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # save image
    
    plt.savefig(save_path+"/"+imagePath.split("/")[-1].split(".")[0]+".jpg")

class DETRDetector():
    def __init__(self):
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = model
        self.transform = transform
        model.eval();
        
    def onImage(self,imagePath):
        # image = cv2.imread(imagePath)
        im = Image.open(imagePath)
        model = self.model

        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0)

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9
        scores = []
        for i in range(len(probas)):
            if keep[i]:
                scores.append(probas[i].max())

        pred_names = [class_names[i] for i in probas.argmax(-1)[keep]]
        # if there is no folder named detr_output, create one
        if not os.path.exists("detr_output"):
            os.makedirs("detr_output")
        # we will have a folder for each image in the detr_output folder
        # if there is no folder named detr_output/image_name, create one. If there is, add _1, _2, _3, etc.
        if not os.path.exists("detr_output/"+imagePath.split("/")[-1].split(".")[0]):
            save_path = "detr_output/"+imagePath.split("/")[-1].split(".")[0]
            os.makedirs(save_path)
        else:
            i = 1
            while os.path.exists("detr_output/"+imagePath.split("/")[-1].split(".")[0]+"_"+str(i)):
                i += 1

            save_path = "detr_output/"+imagePath.split("/")[-1].split(".")[0]+"_"+str(i)
            os.makedirs(save_path)

        coordinates = outputs['pred_boxes'][0, keep].cpu().numpy()
        for i in range(len(coordinates)):
            coordinates[i][0] = coordinates[i][0]*im.size[1]
            coordinates[i][1] = coordinates[i][1]*im.size[0]
            coordinates[i][2] = coordinates[i][2]*im.size[1]
            coordinates[i][3] = coordinates[i][3]*im.size[0]
        
        label_id = 0
        temp = []
        coordinates = np.around(coordinates, decimals=1)
        print("")
        # print(coordinates)
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
            
            
            name = str(pred_names[i]).lower()
            print(name)
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

            temp.append(np.around(scores[i].item(),decimals=4))

            label_id += 1
            # convert to string
            temp = [str(x) for x in temp]
            # join list items by comma
            temp = ','.join(temp)

            print("--------------------")
            print(temp)

            print(save_path)
            print("--------------------")

            with open(save_path+"/"+"labels.txt", "a") as f:
                f.write(temp)
                f.write("\n")

            temp = []  # clear list


        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        plot_results(im, probas[keep], bboxes_scaled,imagePath,save_path)