import sys
import os
import time
import glob
import pandas as pd
from datetime import datetime



yoloPath = "./yolov7/yolov7"
detectorPath = "./detectron2/detectron2"
detrPath = "./detr/detr"
prbnetPath = "./prbnet/PRBNet_PyTorch/prb"
maskdinoPath = "./MaskDINO/demo"
outputPath = "./main/output"

sys.path.append(yoloPath)
sys.path.append(detectorPath)
sys.path.append(detrPath)
sys.path.append(prbnetPath)
print(os.getcwd())


from Detector import *
from DETRDetector import *


class ModelRunner:
    def __init__(self,root):
        self.root = root
        self.imagePath = None
        self.modelName = None
        self.inputfolder = None
        self.start_time = 0
        self.end_time = 0

    def run_on_image(self):
        if self.modelName == "MaskDino - ade20k":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            image_name = self.imagePath.split("/")[-1].split(".")[0]
            os.system("python3" + " " + maskdinoPath + "/demo.py" + " " + "--output" + " " + os.getcwd() + "/" + outputPath + "/maskdino-ade20k_output_" + image_name + "_" + tag + " " + "--config-file" + " " + maskdinoPath + "/../configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml" + " " + "--input" + " " + self.imagePath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../models/ade20k_48.7miou.pth")
            # print("Output saved in " + outputPath + "/maskdino-ade20k_output_" + tag)
            self.root.after_segmentation(outputPath + "/maskdino-ade20k_output_" + image_name+ "_" + tag + ".png")
        
        elif self.modelName == "MaskDino - cityscapes":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            image_name = self.imagePath.split("/")[-1].split(".")[0]
            os.system("python3" + " " + maskdinoPath + "/demo.py" + " " + "--output" + " " + os.getcwd() + "/" + outputPath + "/maskdino-cityscapes_output_" + image_name + "_" + tag + " " + "--config-file" + " " + maskdinoPath + "/../configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml" + " " + "--input" + " " + self.imagePath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth")
            # print("Output saved in " + outputPath + "/maskdino-cityscapes_output_" + tag + ".png")
            self.root.after_segmentation(outputPath + "/maskdino-cityscapes_output_" + image_name + "_" + tag + ".png")

        elif self.modelName == "yolov7":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            # os.system("python3" + " " + yoloPath + "/detect.py" + " " + "--weights" + " " + yoloPath + "/yolov7-e6e.pt" + " " + "--conf" + " " + "0.60" + " " + "--img-size" + " " + "640" + " " + "--source" + " " + self.imagePath + " --save-txt" + " --no-trace" + " --save-img" + " --save-img-path " + os.getcwd() + "/" + outputPath + "/yolov7_output_" + tag + ".png")
            os.system("python3" + " " + yoloPath + "/detect.py" + " " + "--weights" + " " + yoloPath + "/yolov7-e6e.pt" + " " + "--conf" + " " + "0.60" + " " + "--img-size" + " " + "640" + " " + "--source" + " " + self.imagePath + " --save-txt" + " --no-trace" + " --save-csv-path " + os.getcwd() + "/" + outputPath + "/yolov7_output_" + tag + ".csv")
            
            # print("Output saved in " + outputPath + "/yolov7_output_" + tag + ".png")
            if os.path.exists(os.getcwd() + "/" + outputPath + "/yolov7_output_" + tag + ".csv"):
                labels = pd.read_csv(os.getcwd() + "/" + outputPath + "/yolov7_output_" + tag + ".csv", header=None)
                labels.columns =['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']
                print("Read Labels : ")
                print(labels)
                self.root.load_one_image_labels(labels)
            else:
                # return an empty dataframe
                self.root.load_one_image_labels(pd.DataFrame(columns=['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']))

            # self.root.update_prediction(outputPath + "/yolov7_output_" + tag + ".png")

        elif self.modelName == "detectron2":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            detector = Detector()
            start_time = time.time()
            detector.onImage(self.imagePath, save_img=True, save_img_path=os.getcwd() + "/" + outputPath + "/detectron2_output_" + tag + ".csv",model_name="detectron2")
            end_time = time.time()
            # print("Time taken for detectron2: ", end_time - start_time)
            if os.path.exists(os.getcwd() + "/" + outputPath + "/detectron2_output_" + tag + ".csv"):
                labels = pd.read_csv(os.getcwd() + "/" + outputPath + "/detectron2_output_" + tag + ".csv", header=None)
                labels.columns =['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']
                print("Read Labels : ")
                print(labels)
                self.root.load_one_image_labels(labels)
            else:
                # return an empty dataframe
                self.root.load_one_image_labels(pd.DataFrame(columns=['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']))

            # self.root.update_prediction(outputPath + "/detectron2_output_" + tag + ".png")
       
        elif self.modelName == "detr":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            detector = DETRDetector()
            start_time = time.time()
            detector.onImage(self.imagePath,save_csv_path=os.getcwd() + "/" + outputPath + "/detr_output_" + tag + ".csv",model_name="detr")
            end_time = time.time()
            # print("Time taken for detr: ", end_time - start_time)
            # read csv 
            if os.path.exists(os.getcwd() + "/" + outputPath + "/detr_output_" + tag + ".csv"):
                labels = pd.read_csv(os.getcwd() + "/" + outputPath + "/detr_output_" + tag + ".csv", header=None)
                labels.columns =['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']
                print("Read Labels : ")
                print(labels)
                self.root.load_one_image_labels(labels)
            else:
                # return an empty dataframe
                self.root.load_one_image_labels(pd.DataFrame(columns=['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']))
            
            # self.root.update_prediction(outputPath + "/detr_output_" + tag + ".jpg")

        elif self.modelName == "prbnet":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            current_dir = os.getcwd()
            os.chdir(prbnetPath)
            command = "python3 detect.py --weights yolov7-prb.pt --conf 0.60 --img-size 640 --source " + self.imagePath + " --no-trace --save-txt --nosave --save-csv-path " + current_dir + "/" + outputPath + "/prbnet_output_" + tag + ".csv"
            start_time = time.time() 
            os.system(command)
            os.chdir("./../../../")
            end_time = time.time()
            
            if os.path.exists(os.getcwd() + "/" + outputPath + "/prbnet_output_" + tag + ".csv"):
                labels = pd.read_csv(os.getcwd() + "/" + outputPath + "/prbnet_output_" + tag + ".csv", header=None)
                labels.columns =['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']
                print("Read Labels : ")
                print(labels)
                self.root.load_one_image_labels(labels)
            else:
                # return an empty dataframe
                self.root.load_one_image_labels(pd.DataFrame(columns=['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']))

    def run_on_folder(self):
        
        if self.modelName == "MaskDino - ade20k":
            # create output folder
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            folderOutputPath = os.getcwd() + "/main/output/maskdino-ade20k_output_" + tag
            os.mkdir(folderOutputPath)
            os.system("python3 " + maskdinoPath + "/demo.py " + "--config-file " + maskdinoPath + "/../configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml " + "--input " + self.inputfolder + "/*.png" + " --output " + folderOutputPath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../models/ade20k_48.7miou.pth")

        elif self.modelName == "MaskDino - cityscapes":
            # create output folder
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            folderOutputPath = os.getcwd() + "/main/output/maskdino-cityscapes_output_" + tag
            os.mkdir(folderOutputPath)
            os.system("python3 " + maskdinoPath + "/demo.py " + "--config-file " + maskdinoPath + "/../configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml " + "--input " + self.inputfolder + "/*.png" + " --output " + folderOutputPath + " --opts MODEL.WEIGHTS " + maskdinoPath + "/../maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth")            
        
        elif self.modelName == "yolov7":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            # start_time = time.time() 
            
            os.system("python3" + " " + yoloPath + "/detect.py" + " " + "--weights" + " " + yoloPath + "/yolov7-e6e.pt" + " " + "--conf" + " " + "0.60" + " " + "--img-size" + " " + "640" + " " + "--source" + " " + self.inputfolder + " --save-txt --no-trace --folder")

            files = glob.glob("yolov7_output/exp/*.txt")
            folderOutputPath = os.getcwd() + "/main/output/yolov7_output_" + tag
            os.mkdir(folderOutputPath)
            df = pd.concat([pd.read_csv(f, sep=",",header=None) for f in files])
            df.to_csv( folderOutputPath + "/yolov7_output_" + tag + ".csv", index=False, header=False)
            # remove yolov7_output folder
            os.system("rm -rf yolov7_output")
        
        elif self.modelName == "detectron2":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            folderOutputPath = os.getcwd() + "/main/output/detectron2_output_" + tag
            os.mkdir(folderOutputPath)
            # start_time = time.time()
            detector = Detector()
            start_time = time.time()

            # for all images in the folder
            for f in glob.glob(self.inputfolder + "/*"):
                # if jpg or png or jpeg
                if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                    detector.onImage(f)

            folders = glob.glob("detectron2_output/*")

            df = pd.DataFrame()
            # open each folder and read txt file and append to csv
            for folder in folders:
                for f in glob.glob(folder + "/*.txt"):
                    df = pd.concat([df, pd.read_csv(f, sep=",",header=None)])
        
            
            df.to_csv( folderOutputPath + "/detectron2_output_" + tag + ".csv", index=False, header=False)

            # remove detectron2_output folder
            os.system("rm -rf detectron2_output")
            print("--- Total time : %s seconds ---" % (time.time() - start_time))
            # print("--- Total time : %s seconds ---" % (time.time() - start_time))

        elif self.modelName == "detr":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            folderOutputPath = os.getcwd() + "/main/output/detr_output_" + tag
            os.mkdir(folderOutputPath)
            detector = DETRDetector()

            start_time = time.time()
            # for all images in the folder
            for f in glob.glob(self.inputfolder + "/*"):
                # if jpg or png or jpeg
                if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                    detector.onImage(f)

            folders = glob.glob("detr_output/*")

            df = pd.DataFrame()
            # open each folder and read txt file and append to csv
            for folder in folders:
                for f in glob.glob(folder + "/*.txt"):
                    df = pd.concat([df, pd.read_csv(f, sep=",",header=None)])

            df.to_csv( folderOutputPath + "/detr_output_" + tag + ".csv", index=False, header=False)
                
            # remove detectron2_output folder
            # os.system("rm -rf detr_output")
            print("--- Total time : %s seconds ---" % (time.time() - start_time))

        elif self.modelName == "prbnet":
            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year
            current_time = datetime.now().strftime("%H-%M-%S")
            tag = str(currentDay) + "-" + str(currentMonth) + "-" + str(currentYear) + "_" + current_time
            folderOutputPath = os.getcwd() + "/main/output/prbnet_output_" + tag
            os.mkdir(folderOutputPath)
            os.chdir(prbnetPath)
            
            command = "python3 detect.py --weights yolov7-prb.pt --conf 0.60 --img-size 640 --source " + self.inputfolder + " --save-txt --nosave --folder --no-trace"
            # print(command)
            
            start_time = time.time()
            
            os.system(command)

            os.chdir("./../../../")

            files = glob.glob("prbnet_output/exp/*.txt")

            df = pd.concat([pd.read_csv(f, sep=",",header=None) for f in files])
            
            df.to_csv( folderOutputPath + "/prbnet_output_" + tag + ".csv", index=False, header=False)
            
            # remove prbnet_output folder
            os.system("rm -rf prbnet_output")

            print("--- Total time : %s seconds ---" % (time.time() - start_time))