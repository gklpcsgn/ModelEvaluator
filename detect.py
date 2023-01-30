import sys
import os
import time
import glob
import pandas as pd

yoloPath = "./yolov7/yolov7"
detectorPath = "./detectron2/detectron2"
detrPath = "./detr/detr"
prbnetPath = "./prbnet/PRBNet_PyTorch/prb"

sys.path.append(yoloPath)
sys.path.append(detectorPath)
sys.path.append(detrPath)
sys.path.append(prbnetPath)

# print(sys.path)

# Ask yolo or detectron2 to detect objects in the image

print("Please write the path to the image you want to detect objects in:")
imagePath = input()

print("Please write the model name you want to use [yolo/detectron2/detr/prbnet]:")
modelName = input()

if modelName == "yolo":
    start_time = time.time()
    # os.system("conda activate yolov7")
    os.system("python3" + " " + yoloPath + "/detect.py" + " " + "--weights" + " " + yoloPath + "/yolov7-e6e.pt" + " " + "--conf" + " " + "0.60" + " " + "--img-size" + " " + "640" + " " + "--source" + " " + imagePath + " --save-txt" + " --no-trace")


    # remove if output.csv exists
    if os.path.exists(imagePath + "/output.csv"):
        os.remove(imagePath + "/output.csv")

    glob = glob.glob("yolov7_output/exp/*.txt")

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
    os.system("rm -rf detr_output")
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
else:
    print("Please write a valid model name")
