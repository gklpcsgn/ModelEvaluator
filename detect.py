import sys
import os
import time
import glob
import pandas as pd

yoloPath = "./yolov7/yolov7"
detectorPath = "./detectron2/detectron2"
detrPath = "./detr/detr"
prbnetPath = "./prbnet/PRBNet_PyTorch/prb"
maskdinoPath = "./MaskDINO/demo"

sys.path.append(yoloPath)
sys.path.append(detectorPath)
sys.path.append(detrPath)
sys.path.append(prbnetPath)

# print(sys.path)

# Ask yolo or detectron2 to detect objects in the image

print("Please write the path to the image you want to detect objects in:")
imagePath = input()
print("Please write the model name you want to use [yolo/detectron2/detr/prbnet/maskdino]:")
modelName = input()

inputfolder = "/".join(imagePath.split("/")[:-1])

if modelName == "maskdino":
    dataset = input("Please write the dataset name you want to use [ade20k/coco/cityscapes]:")
    if dataset == "ade20k":
        os.system("python3" + " " + maskdinoPath + "/demo.py" + " " + "--output" + " " + inputfolder + "/maskdino_output" + " " + "--config-file" + " " + maskdinoPath + "/../configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml" + " " + "--input" + " " + imagePath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../models/ade20k_48.7miou.pth")

    elif dataset == "cityscapes":
        os.system("python3" + " " + maskdinoPath + "/demo.py" + " " + "--output" + " " + inputfolder + "/maskdino_output" + " " + "--config-file" + " " + maskdinoPath + "/../configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml" + " " + "--input" + " " + imagePath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth")

    print("Output saved in " + inputfolder + "/maskdino_output")


elif modelName == "yolo":
    start_time = time.time()
    # os.system("conda activate yolov7")
    os.system("python3" + " " + yoloPath + "/detect.py" + " " + "--weights" + " " + yoloPath + "/yolov7-e6e.pt" + " " + "--conf" + " " + "0.60" + " " + "--img-size" + " " + "640" + " " + "--source" + " " + imagePath + " --save-txt" + " --no-trace")
    # remove if output.csv exists
    if os.path.exists(imagePath + "/output.csv"):
        os.remove(imagePath + "/output.csv")

    glob = glob.glob("yolov7_output/exp/*.csv")

    df = pd.concat([pd.read_csv(f, sep=",",header=None) for f in glob])
    df.to_csv( imagePath + "/yolo_output.csv", index=False, header=False)
    # remove yolov7_output folder
    os.system("rm -rf yolov7_output")

    print("--- Total time : %s seconds ---" % (time.time() - start_time))


elif modelName == "detectron2":
    # os.system("conda activate detectron2")
    from Detector import *
    detector = Detector()
    start_time = time.time()

    # for all images in the folder
    for f in glob.glob(imagePath + "/*"):
        # if jpg or png or jpeg
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
            detector.onImage(f)
    
    # remove if output.csv exists
    if os.path.exists(imagePath + "/output.csv"):
        os.remove(imagePath + "/output.csv")

    f_glob = glob.glob("detectron2_output/*")

    df = pd.DataFrame()
    # open each folder and read txt file and append to csv
    for folder in f_glob:
        for f in glob.glob(folder + "/*.txt"):
            df = pd.concat([df, pd.read_csv(f, sep=",",header=None)])
    print(imagePath)
    df.to_csv( imagePath + "/detectron2_output.csv", index=False, header=False)
           

    # remove detectron2_output folder
    os.system("rm -rf detectron2_output")
    print("--- Total time : %s seconds ---" % (time.time() - start_time))
elif modelName == "detr":


    # TODO: WORKS SOOOOO SLOW  *10 min for 456 img.


    # os.system("conda activate detr")
    from DETRDetector import *
    detector = DETRDetector()

    start_time = time.time()
 # for all images in the folder
    for f in glob.glob(imagePath + "/*"):
        # if jpg or png or jpeg
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
            detector.onImage(f)
    
    # remove if output.csv exists
    if os.path.exists(imagePath + "/output.csv"):
        os.remove(imagePath + "/output.csv")

    f_glob = glob.glob("detr_output/*")

    df = pd.DataFrame()
    # open each folder and read txt file and append to csv
    for folder in f_glob:
        for f in glob.glob(folder + "/*.txt"):
            df = pd.concat([df, pd.read_csv(f, sep=",",header=None)])

    df.to_csv( imagePath + "/detr_output.csv", index=False, header=False)
           

    # remove detectron2_output folder
    # os.system("rm -rf detr_output")
    print("--- Total time : %s seconds ---" % (time.time() - start_time))

elif modelName == "prbnet":
    # os.system("conda activate prbnet")
    # os.system("ls")
    # print(prbnetPath + "/yolov7-prb.pt")

    os.chdir(prbnetPath)
    command = "python3 detect.py --weights yolov7-prb.pt --conf 0.60 --img-size 640 --source " + imagePath + " --save-txt --nosave"
    # print(command)
    
    start_time = time.time()
    
    os.system(command)

    os.chdir("./../../../")

    # remove if output.csv exists
    if os.path.exists(imagePath + "/output.csv"):
        os.remove(imagePath + "/output.csv")

    glob = glob.glob("prbnet_output/exp/*.txt")

    df = pd.concat([pd.read_csv(f, sep=",",header=None) for f in glob])
    df.to_csv( imagePath + "/prbnet_output.csv", index=False, header=False)
    # remove prbnet_output folder
    os.system("rm -rf prbnet_output")

    print("--- Total time : %s seconds ---" % (time.time() - start_time))

elif modelName == "MaskDino - ade20k":
    folderOutputPath = os.getcwd() + "/maskdino-ade20k_output/"
    os.mkdir(folderOutputPath)
    os.system("python3 " + maskdinoPath + "/demo.py " + "--config-file " + maskdinoPath + "/../configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml " + "--input " + imagePath + "/*.png" + " --no-image-output --output " + folderOutputPath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../models/ade20k_48.7miou.pth")

elif modelName == "MaskDino - cityscapes":

    folderOutputPath = os.getcwd() + "/maskdino-cityscapes_output/"
    os.mkdir(folderOutputPath)            
    os.system("python3 " + maskdinoPath + "/demo.py " + "--config-file " + maskdinoPath + "/../configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml " + "--input " + imagePath + "/*.png" + " --no-image-output --output " + folderOutputPath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth")

else:
    print("Please write a valid model name")


