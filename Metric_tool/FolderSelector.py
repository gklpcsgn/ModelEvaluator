import json
import sys
import tkinter as tk
from tkinter import HORIZONTAL, RAISED, SUNKEN, OptionMenu, Scale, ttk
from tkinter import Frame, Toplevel, StringVar, Button, Event, Label, PhotoImage, Scrollbar, filedialog, image_names, simpledialog,Entry,W
import os
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from metric_functions import *
from metric_functions import get_metric_results
import pandas as pd
class MyImage:
    def __init__(self,image_name,imageID):
        self.imageID = imageID
        self.image_name = image_name
        self.class_based_results = None
        self.ground_truths = pd.DataFrame(columns=["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class"])
        self.detections = pd.DataFrame(columns=["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"])
        self.original_ground_truths = pd.DataFrame(columns=["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class"])
        self.original_detections = pd.DataFrame(columns=["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"])
        
        self.gt = None
        self.tp = None
        self.fp = None
        self.fn = None
        self.precision = None
        self.recall = None
        self.map = None

    def toString(self):
        # id | image name | ground truth tp | ground truth fp | ground truth fn | detection tp | detection fp | detection fn | precision | recall | map
        # image name must be 25 characters long
        # all other values must be 5 characters long
        # all values must be separated by 4 spaces
        # all values must be rounded to 2 decimal places
        # all values must be converted to strings
        # all values must be middle aligned
        id = str(self.imageID) if self.imageID is not None else ""
        name = self.image_name if self.image_name is not None else ""
        gt = self.ground_truths.shape[0] if self.ground_truths is not None else ""
        tp = str(int(self.tp)) if self.tp is not None else ""
        fp = str(int(self.fp)) if self.fp is not None else ""
        fn = str(int(self.fn)) if self.fn is not None else ""
        precision = str(np.around(self.precision,2)) if self.precision is not None else ""
        recall = str(np.around(self.recall,2)) if self.recall is not None else ""
        map = str(np.around(self.map,2)) if self.map is not None else ""

        # get first 25 characters of name
        if len(name) > 25:
            name = name[:22] + "..."

        out = "{:<5} {:<30} {:<6} {:<6} {:<6} {:<6} {:<9}      {:<6}      {:<3}".format(id,name,gt,tp,fp,fn,precision,recall,map)
        
        return out

    def set_metrics(self,result):
        self.class_based_results = result["class_based"]
        self.class_based_results.reset_index(inplace=True)
        self.class_based_results.rename(columns={"index":"class"},inplace=True)

        self.overall_results = result["overall"]
        self.map_results = result["map"]

        self.tp = self.overall_results["TP"]
        self.fp = self.overall_results["FP"]
        self.fn = self.overall_results["FN"]
        self.precision = self.overall_results["precision"]
        self.recall = self.overall_results["recall"]
        self.map = self.map_results["AP"].iloc[2]

    def set_zero_metrics(self):
        self.class_based_results = None
        self.overall_results = None
        self.map_results = None

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.map = 0
        
class FolderSelector:

    def __init__(self,root):
        self.root = root
        self.root.title("TyX Technologies Metric Calculation Tool")
        self.root.geometry("1400x900")
        self.current_folder = None
        self.current_image = None
        self.scale = 0
        self.output_path = None        
        self.detection_df = None
        self.ground_truth_df = None
        self.detection_file = None
        self.ground_truth_file = None
        self.catagories = None
        self.my_images = []
        self.image_dict = {}
        self.name_to_id = {}


        self.show_ground_truths_flag = False
        self.show_detections_flag = False
        self.show_all_flag = True
        self.show_car_flag = False
        self.show_truck_flag = False
        self.show_pedestrian_flag = False
        self.show_bus_flag = False
        self.show_animal_flag = False
        self.show_others_flag = False
        self.folder_selected_flag = False
        self.detection_file_selected_flag = False
        self.ground_truth_file_selected_flag = False


        self.car_threshold = 0.5
        self.truck_threshold = 0.5
        self.pedestrian_threshold = 0.5
        self.bus_threshold = 0.5
        self.animal_threshold = 0.5
        self.others_threshold = 0.5    

        self.initUI()

    def initUI(self):

        self.loadJSONCatagories()

        self.image_frame = Frame(self.root,width=896,height=512,padx=10,pady=10)
        self.image_frame.grid(row=0, column=0,rowspan=4)

        self.canvas = tk.Canvas(self.image_frame,width=896,height=512)
        self.canvas.grid(row=0, column=0)

        self.scroll_x = tk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")

        self.scroll_y = tk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.image_frame.grid_rowconfigure(0, weight=1) 
        self.image_frame.grid_columnconfigure(0, weight=1)
        # self.image_frame.grid_rowconfigure(2, minsize=100)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.configure(yscrollincrement='1')
        self.canvas.configure(xscrollincrement='1')
        self.canvas.configure(xscrollcommand=self.scroll_x.set)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvasImage = self.canvas.create_image(0, 0, anchor="nw",image=None)

        self.button_frame = Frame(self.root,padx=10,pady=10)
        self.button_frame.grid(row=0, column=1,sticky="n")

        self.select_folder_button = Button(self.button_frame, text="Select Folder", command=self.select_folder_button_click)
        self.select_folder_button.grid(row=0, column=0,sticky="new")

        self.select_ground_truth_button = Button(self.button_frame, text="Select Ground Truth", command=self.select_ground_truth_button_click)
        self.select_ground_truth_button.grid(row=0, column=1,sticky="new")
        self.select_ground_truth_button.config(state="disabled")

        self.select_detection_button = Button(self.button_frame, text="Select Detection", command=self.select_detection_button_click)
        self.select_detection_button.grid(row=0, column=2,sticky="new")
        self.select_detection_button.config(state="disabled")

        self.show_ground_truth_button = Button(self.button_frame, text="Show Ground Truth", command=self.show_ground_truth_button_click)
        self.show_ground_truth_button.grid(row=1, column=0,sticky="new")
        self.show_ground_truth_button.config(state="disabled")

        self.show_detection_button = Button(self.button_frame, text="Show Detection", command=self.show_detection_button_click)
        self.show_detection_button.grid(row=1, column=1,sticky="new")
        self.show_detection_button.config(state="disabled")
        
        self.calculate_metrics_button = Button(self.button_frame, text="Calculate Metrics", command=self.calculate_metrics_button_click)
        self.calculate_metrics_button.grid(row=1, column=2,sticky="new")
        self.calculate_metrics_button.config(state="disabled")

        # add a frame to show class based metrics
        self.class_metrics_frame = Frame(self.root,padx=10,pady=10)
        self.class_metrics_frame.grid(row=1, column=1,sticky="n")

        self.class_metrics_frame.grid_rowconfigure(0, weight=1)
        self.class_metrics_frame.grid_columnconfigure(0, weight=1)
        self.class_metrics_frame.grid_rowconfigure(2, minsize=100)

        self.class_metrics_header = tk.Label(self.class_metrics_frame, text="Class Metrics\n")
        self.class_metrics_header.grid(row=0, column=0, sticky="we")

        class_metrics_column_text = "{:<10} {:<6} {:<6} {:<6} {:<9}      {:<6}".format("Class","TP","FP","FN","Precision","Recall")
        self.class_metrics_columns = tk.Label(self.class_metrics_frame, text=class_metrics_column_text)
        self.class_metrics_columns.grid(row=1, column=0, sticky="w")

        self.class_metrics_list = tk.Listbox(self.class_metrics_frame, height=20, width=50)
        self.class_metrics_list.grid(row=2, column=0, sticky="nsew")

        # add a scroll bar to the class metrics list
        self.class_metrics_scroll = tk.Scrollbar(self.class_metrics_frame, orient="vertical", command=self.class_metrics_list.yview)
        self.class_metrics_scroll.grid(row=2, column=1, sticky="ns")
        self.class_metrics_list.configure(yscrollcommand=self.class_metrics_scroll.set)

        # We will add a overall metrics listbox
        
        self.overall_metrics_header = tk.Label(self.class_metrics_frame, text="\n\nOverall Metrics\n")
        self.overall_metrics_header.grid(row=3, column=0, sticky="we")

        # We will create a dynamic label for each overall metric [0.5AP, 0.75AP, mAP]
        
        zero_five_AP_label_text = "{:<8} : ".format("0.5 AP")
        self.zero_five_AP_label = tk.Label(self.class_metrics_frame, text=zero_five_AP_label_text)
        self.zero_five_AP_label.grid(row=4, column=0, sticky="w")

        zero_seven_five_AP_label_text = "{:<8} : ".format("0.75 AP")
        self.zero_seven_five_AP_label = tk.Label(self.class_metrics_frame, text=zero_seven_five_AP_label_text)
        self.zero_seven_five_AP_label.grid(row=5, column=0, sticky="w")

        map_label_text = "{:<8} : ".format("mAP")
        self.mAP_label = tk.Label(self.class_metrics_frame, text=map_label_text)
        self.mAP_label.grid(row=6, column=0, sticky="w")

                

        # we will have a listbox to show the images in the current folder
        self.image_list_frame = Frame(self.root,width=512,height=384,padx=10,pady=10)
        self.image_list_frame.grid(row=4, column=0, padx=10,pady=10,sticky="new")

        self.image_list_frame.grid_rowconfigure(0, weight=1)
        self.image_list_frame.grid_columnconfigure(0, weight=1)
        header = "{:<5} {:<30} {:<6} {:<6} {:<6} {:<6} {:<9}      {:<6}      {:<3}".format("ID","Image Name","GT","TP","FP","FN","Precision","Recall","map")
        self.imageListHeader = tk.Label(self.image_list_frame, text=header)
        self.imageListHeader.grid(row=0, column=0, sticky="w",columnspan=4)
        
        self.imageList = tk.Listbox(self.image_list_frame, height = 25)
        self.imageList.grid(row=1, column=0, sticky="new",columnspan=4)
        self.imageList.bind('<<ListboxSelect>>', self.on_image_select)
        
        # add a scrollbar to the listbox
        self.scrollbar = Scrollbar(self.image_list_frame, orient="vertical",command=self.imageList.yview)
        self.scrollbar.grid(row=1, column=5, sticky="nsew")
        self.imageList.config(yscrollcommand=self.scrollbar.set)


        # create a frame to put sort by dropdown list
        self.sort_by_frame = Frame(self.root,padx=10,pady=10)
        self.sort_by_frame.grid(row=4, column=1,sticky="n")

        # we will put a dropdown list to sort the images
        self.sort_by_label = tk.Label(self.sort_by_frame, text="Sort By : ")
        self.sort_by_label.grid(row=0, column=0, sticky="w")
        self.sort_by = StringVar(self.root)
        self.sort_by.set("Image Name") # default value
        self.sort_by_dropdown = OptionMenu(self.sort_by_frame, self.sort_by, "Image Name (DESC)", "Image Name (ASC)", "TP (DESC)", "TP (ASC)", "FP (DESC)", "FP (ASC)", "FN (DESC)", "FN (ASC)", "Precision (DESC)", "Precision (ASC)", "Recall (DESC)", "Recall (ASC)", "map (DESC)", "map (ASC)")
        self.sort_by_dropdown.configure(state= 'disabled')
        self.sort_by_dropdown.grid(row=0, column=1, sticky="w")
        self.sort_by.trace("w", self.sort_images)

        # Thresholds and Filters Label
        self.thresholds_and_filters_label = tk.Label(self.sort_by_frame, text="\n\nThresholds and Filters\n")
        self.thresholds_and_filters_label.grid(row=1, column=0,columnspan=3, sticky="we")
        # Add buttons for All,Car,Truck,Bus,Pedestrian,Animal,Others. There will be sliding bars for each class to set the threshold
        # set all to disabled by default
        self.all_button = Button(self.sort_by_frame, text="All", command=self.all_button_click)
        self.all_button.grid(row=2, column=0, sticky="sew")
        self.all_button.configure(state= 'disabled')
        self.all_button.configure(relief="sunken")
        self.car_button = Button(self.sort_by_frame, text="Car", command=self.car_button_click)
        self.car_button.grid(row=3, column=0, sticky="sew")
        self.car_button.configure(state= 'disabled')
        self.truck_button = Button(self.sort_by_frame, text="Truck", command=self.truck_button_click)
        self.truck_button.grid(row=4, column=0, sticky="sew")
        self.truck_button.configure(state= 'disabled')
        self.bus_button = Button(self.sort_by_frame, text="Bus", command=self.bus_button_click)
        self.bus_button.grid(row=5, column=0, sticky="sew")
        self.bus_button.configure(state= 'disabled')
        self.pedestrian_button = Button(self.sort_by_frame, text="Pedestrian", command=self.pedestrian_button_click)
        self.pedestrian_button.grid(row=6, column=0, sticky="sew")
        self.pedestrian_button.configure(state= 'disabled')
        self.animal_button = Button(self.sort_by_frame, text="Animal", command=self.animal_button_click)
        self.animal_button.grid(row=7, column=0, sticky="sew")
        self.animal_button.configure(state= 'disabled')
        self.others_button = Button(self.sort_by_frame, text="Others", command=self.others_button_click)
        self.others_button.grid(row=8, column=0, sticky="sew")
        self.others_button.configure(state= 'disabled')

        # put sliders next to the buttons
        # self.all_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.all_slider_change)
        # self.all_slider.grid(row=2, column=1, sticky="nsew")
        self.car_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.car_slider_change)
        self.car_slider.grid(row=3, column=2,columnspan=2, sticky="nsew")
        self.truck_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.truck_slider_change)
        self.truck_slider.grid(row=4, column=2,columnspan=2, sticky="nsew")
        self.bus_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.bus_slider_change)
        self.bus_slider.grid(row=5, column=2,columnspan=2, sticky="nsew")
        self.pedestrian_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.pedestrian_slider_change)
        self.pedestrian_slider.grid(row=6, column=2,columnspan=2, sticky="nsew")
        self.animal_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.animal_slider_change)
        self.animal_slider.grid(row=7, column=2,columnspan=2, sticky="nsew")
        self.others_slider = Scale(self.sort_by_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, command=self.others_slider_change)
        self.others_slider.grid(row=8, column=2,columnspan=2, sticky="nsew")

        # self.all_slider.set(0.5)
        self.car_slider.set(0.5)
        self.truck_slider.set(0.5)
        self.bus_slider.set(0.5)
        self.pedestrian_slider.set(0.5)
        self.animal_slider.set(0.5)
        self.others_slider.set(0.5)


    def all_button_click(self):
        if self.show_all_flag is False:
            self.show_all_flag = True
            self.all_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_all_flag = False
            self.all_button.configure(relief="raised")
            self.filter_image_list()
        
    def car_button_click(self):
        if self.show_car_flag is False:
            self.show_car_flag = True
            self.car_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_car_flag = False
            self.car_button.configure(relief="raised")
            self.filter_image_list()
    def truck_button_click(self):
        if self.show_truck_flag is False:
            self.show_truck_flag = True
            self.truck_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_truck_flag = False
            self.truck_button.configure(relief="raised")
            self.filter_image_list()
    def bus_button_click(self):
        if self.show_bus_flag is False:
            self.show_bus_flag = True
            self.bus_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_bus_flag = False
            self.bus_button.configure(relief="raised")
            self.filter_image_list()
    def pedestrian_button_click(self):
        if self.show_pedestrian_flag is False:
            self.show_pedestrian_flag = True
            self.pedestrian_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_pedestrian_flag = False
            self.pedestrian_button.configure(relief="raised")
            self.filter_image_list()
    def animal_button_click(self):
        if self.show_animal_flag is False:
            self.show_animal_flag = True
            self.animal_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_animal_flag = False
            self.animal_button.configure(relief="raised")
            self.filter_image_list()
    def others_button_click(self):  
        if self.show_others_flag is False:
            self.show_others_flag = True
            self.others_button.configure(relief="sunken")
            self.filter_image_list()
        else:
            self.show_others_flag = False
            self.others_button.configure(relief="raised")
            self.filter_image_list()

    def car_slider_change(self,value):
        self.car_threshold = float(value)
    def truck_slider_change(self,value):
        self.truck_threshold = float(value)
    def bus_slider_change(self,value):
        self.bus_threshold = float(value)
    def pedestrian_slider_change(self,value):
        self.pedestrian_threshold = float(value)
    def animal_slider_change(self,value):
        self.animal_threshold = float(value)
    def others_slider_change(self,value):
        self.others_threshold = float(value)

    def filter_image_list(self):
        # We will filter the image list based on the flags
        self.imageList.delete(0,tk.END)
        for i in range(len(self.my_images)):
            if self.show_all_flag:
                self.imageList.insert(tk.END, self.my_images[i].toString())
            else:
                my_image = self.my_images[i]
                if my_image.detections.shape[0] == 0:
                    continue
                detections = my_image.detections
                # for each row in detections, check if the class is present in the class list
                for index, detection in detections.iterrows():
                    if detection["class"] == 3 and self.show_car_flag:
                        self.imageList.insert(tk.END, my_image.toString())
                        break
                    elif detection["class"] == 8 and self.show_truck_flag:
                        self.imageList.insert(tk.END, my_image.toString())
                        break
                    elif detection["class"] == 6 and self.show_bus_flag:
                        self.imageList.insert(tk.END, my_image.toString())
                        break
                    elif detection["class"] == 1 and self.show_pedestrian_flag:
                        self.imageList.insert(tk.END, my_image.toString())
                        break
                    elif (detection["class"] >= 16 and detection["class"] <= 25 ) and self.show_animal_flag:
                        self.imageList.insert(tk.END, my_image.toString())
                        break
                    elif detection["class"] != 3 and detection["class"] != 8 and detection["class"] != 6 and detection["class"] != 1 and (detection["class"] < 15 or detection["class"] > 25) and self.show_others_flag:
                        print("Others added : " + str(detection["class"]))
                        self.imageList.insert(tk.END, my_image.toString())
                        break
                
    def sort_images(self,*args):
        print("Sorting Images")
        # print args
        sortby = self.sort_by.get()
        # to sort, we will sort my_images list and then update the image list
        reverse = sortby.split("(")[1].split(")")[0] == "DESC"
        sortby = sortby.split("(")[0].strip()
        self.sort_my_images(sortby,reverse)
        self.imageList.delete(0,tk.END)

        for i in range(len(self.my_images)):
            self.imageList.insert(tk.END, self.my_images[i].toString())
        self.filter_image_list()

    def sort_my_images(self,sortby,reverse):
        if sortby == "Image Name":
            self.my_images.sort(key=lambda x: x.image_name)
        elif sortby == "TP":
            self.my_images.sort(key=lambda x: x.tp, reverse=reverse)
        elif sortby == "FP":
            self.my_images.sort(key=lambda x: x.fp, reverse=reverse)
        elif sortby == "FN":
            self.my_images.sort(key=lambda x: x.fn, reverse=reverse)
        elif sortby == "Precision":
            self.my_images.sort(key=lambda x: x.precision, reverse=reverse)
        elif sortby == "Recall":
            self.my_images.sort(key=lambda x: x.recall, reverse=reverse)
        elif sortby == "map":
            self.my_images.sort(key=lambda x: x.mAP, reverse=reverse)
        else:
            print("Invalid Sortby")

    def rearrange_detections(self,image_detections):
        print("Rearranging Detections")
        # if the image_detections is empty, return empty dataframe
        if image_detections.shape[0] == 0:
            return image_detections
        
        for index, detection in image_detections.iterrows():
            if detection["class"] == 3:
                if detection["confidence"] < self.car_threshold:
                    image_detections.drop(index, inplace=True)
            elif detection["class"] == 8:
                if detection["confidence"] < self.truck_threshold:
                    image_detections.drop(index, inplace=True)
            elif detection["class"] == 6:
                if detection["confidence"] < self.bus_threshold:
                    image_detections.drop(index, inplace=True)
            elif detection["class"] == 1:
                if detection["confidence"] < self.pedestrian_threshold:
                    image_detections.drop(index, inplace=True)
            elif detection["class"] >= 16 and detection["class"] <= 25:
                if detection["confidence"] < self.animal_threshold:
                    image_detections.drop(index, inplace=True)
            else:
                if detection["confidence"] < self.others_threshold:
                    image_detections.drop(index, inplace=True)
        return image_detections

        
    def calculate_metrics(self):
        print("Calculating Metrics")
        for image in self.my_images:
            # print("image : ", image.image_name)
            # rearrange the detections based on the threshold values
            image.detections = image.original_detections.copy()
            image.detections = self.rearrange_detections(image.detections)
            result = get_metric_results(image.detections,image.ground_truths)
            if result is not None:
                # print("result : ", result)
                image.set_metrics(result)
            else:
                image.set_zero_metrics()

        self.sort_by_dropdown.configure(state='normal')
        
        self.update_image_list()
        self.update_class_metrics()
        self.update_overall_metrics()
        self.filter_image_list()

    def update_overall_metrics(self):
        splitted_image_names = [i.split('.')[0] for i in self.image_names]

        # remove detections from the dataframe if the image is not present in the current folder
        self.detection_df = self.detection_df[self.detection_df['image_id'].isin(splitted_image_names)]
        # remove ground truths from the dataframe if the image is not present in the current folder
        self.ground_truth_df = self.ground_truth_df[self.ground_truth_df['image_id'].isin(splitted_image_names)]

        # print("detection_df : ", self.detection_df)
        # print("ground_truth_df : ", self.ground_truth_df)
        
        results = get_metric_results(self.detection_df,self.ground_truth_df)
        if results is not None:
            map_df = results['map'].T
            map_df.reset_index(inplace=True,drop=True)
            # print("map_df : ", map_df)
            map_df.columns = ['0.5','0.75','mAP']
            self.zero_five_AP_label.config(text="{:<8} : {:<6.4f}".format("0.5 AP",map_df.at[0,'0.5']))
            self.zero_seven_five_AP_label.config(text="{:<8} : {:<6.4f}".format("0.75 AP",map_df.at[0,'0.75']))
            self.mAP_label.config(text="{:<8} : {:<6.4f}".format("mAP",map_df.at[0,'mAP']))


    def update_class_metrics(self):
        df = pd.DataFrame(columns=['class','TP','FP','FN','Precision','Recall'])
        for image in self.my_images:
            class_metrics = image.class_based_results
            # for every row in the class based metrics
            if class_metrics is None:
                continue
            for index, row in class_metrics.iterrows():
                # check if the class is already present in the dataframe
                if row['class'] in df['class'].values:
                    # if present then update the values
                    df.loc[df['class'] == row['class'],'TP'] += row['TP']
                    df.loc[df['class'] == row['class'],'FP'] += row['FP']
                    df.loc[df['class'] == row['class'],'FN'] += row['FN']
                else:
                    # if not present then add the row to the dataframe
                    df = pd.concat([df,row.to_frame().T],ignore_index=True)

        # calculate the precision and recall
        # if TP is 0 then precision and recall will be 0
        for index, row in df.iterrows():
            if row['TP'] == 0:
                df.loc[index,'Precision'] = 0
                df.loc[index,'Recall'] = 0
            else:
                df.loc[index,'Precision'] = row['TP']/(row['TP'] + row['FP'])
                df.loc[index,'Recall'] = row['TP']/(row['TP'] + row['FN'])

        # append the dataframe to the listbox
        self.class_metrics_list.delete(0,tk.END)
        for index, row in df.iterrows():
            text = "{:<10} {:<6} {:<6} {:<6} {:<9.2f}      {:<6.2f}".format(row['class'],int(row['TP']),int(row['FP']),int(row['FN']),row['Precision'],row['Recall'])
            self.class_metrics_list.insert(tk.END,text)

    def set_image_ground_truths(self):
        print("Setting Ground Truths")
        gt = self.ground_truth_df
        for image in self.my_images:
            image.ground_truths = gt[gt['image_id'] == image.image_name.split('.')[0]]
            image.original_ground_truths = image.ground_truths.copy()
            # print("image : ", image.image_name, " ground truths : ", image.ground_truths)

    def set_image_detections(self):
        print("Setting Detections")
        det = self.detection_df
        for image in self.my_images:
            image.detections = pd.concat([image.detections,det[det['image_id'] == image.image_name.split('.')[0]]])
            image.original_detections = image.detections.copy()
            # print("image : ", image.image_name, " detections : ", image.detections)

    def calculate_metrics_button_click(self):
        self.calculate_metrics()


    def refresh_folder(self):
    # if the folder is already selected then clear the listbox
        self.imageList.delete(0, tk.END)
        self.my_images = []
        self.image_dict = {}
        self.name_to_id = {}
        self.image_names = []


    def refresh_app(self):
        # restart the application
        python = sys.executable
        os.execl(python, python, * sys.argv)

    def select_folder_button_click(self):
        # open the file dialog
        self.current_folder = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Folder")
        
        if len(self.current_folder) == 0:
            # create a popup message if the user does not select a folder
            messagebox.showerror("Error","Please select a folder")
            return  
        
        if self.folder_selected_flag:
            self.refresh_folder()
            self.folder_selected_flag = False

        self.folder_selected_flag = True
        self.image_dict = {}
        self.name_to_id = {}

        # list all files in the folder
        id = 0
        for image_name in os.listdir(self.current_folder):
            myimage = MyImage(image_name,id)
            self.image_dict[str(id)] = myimage
            self.my_images.append(myimage)
            self.name_to_id[image_name] = str(id)
            id += 1

        self.image_names = os.listdir(self.current_folder)

        # print(self.image_names)
        # clear the listbox
        self.imageList.delete(0, tk.END)
        # add the files to the listbox if they are images
        for image_name in self.image_names:
            if image_name.endswith(".jpg") or image_name.endswith(".png"): 
                self.imageList.insert(tk.END, self.image_dict[self.name_to_id[image_name]].toString())
    
        self.select_ground_truth_button.config(state="normal")
        self.select_detection_button.config(state="normal")

        self.show_detection_button.config(state="disabled")
        self.show_ground_truth_button.config(state="disabled")
        self.calculate_metrics_button.config(state="disabled")

        # empty class list
        self.class_metrics_list.delete(0, tk.END)

        self.zero_five_AP_label.config(text="{:<8} :".format("0.5 AP"))
        self.zero_seven_five_AP_label.config(text="{:<8} :".format("0.75 AP"))
        self.mAP_label.config(text="{:<8} :".format("mAP"))
        self.sort_by_dropdown.config(state="disabled")


    def refresh_ground_truths(self):
        self.ground_truth_file_selected_flag = False
        self.ground_truth_df = None
        self.select_ground_truth_button.config(state="normal")
        self.show_ground_truth_button.config(state="disabled")
        self.calculate_metrics_button.config(state="disabled")
        self.class_metrics_list.delete(0, tk.END)
        self.zero_five_AP_label.config(text="{:<8} :".format("0.5 AP"))
        self.zero_seven_five_AP_label.config(text="{:<8} :".format("0.75 AP"))
        self.mAP_label.config(text="{:<8} :".format("mAP"))
        self.sort_by_dropdown.config(state="disabled")
        
        for image in self.my_images:
            image.ground_truths = None


    def select_ground_truth_button_click(self):
        # open the file dialog
        self.ground_truth_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Ground Truth File",filetypes=(("CSV files","*.csv"),("all files","*.*")))
        
        if len(self.ground_truth_file) == 0:
            messagebox.showerror("Error","Please select a ground truth file")
            return
        
        if self.ground_truth_file_selected_flag:
            self.refresh_ground_truths()

        self.ground_truth_df = pd.read_csv(self.ground_truth_file)

        if self.ground_truth_df.shape[1] != 9:
            messagebox.showerror("Error","The ground truth file should have 9 columns")
            self.refresh_ground_truths()
            return

        self.ground_truth_file_selected_flag = True
        
        self.ground_truth_df.columns = ["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class"]
        self.ground_truth_df["image_id"] = self.ground_truth_df["image_id"].values.astype('str')

        # remove class 1001 from the ground truth dataframe
        self.ground_truth_df = self.ground_truth_df[self.ground_truth_df['class'] != 1001]
        self.ground_truth_df = self.ground_truth_df[self.ground_truth_df['class'] != -1]

        self.set_image_ground_truths()
        self.show_ground_truth_button.config(state="normal")
        if self.detection_df is not None:
            self.calculate_metrics_button.config(state="normal")

        self.update_image_list()


    def refresh_detections(self):
        self.detection_file_selected_flag = False
        self.detection_df = None
        self.select_ground_truth_button.config(state="normal")
        self.select_detection_button.config(state="normal")
        self.show_detection_button.config(state="disabled")
        self.calculate_metrics_button.config(state="disabled")
        self.class_metrics_list.delete(0, tk.END)
        self.zero_five_AP_label.config(text="{:<8} :".format("0.5 AP"))
        self.zero_seven_five_AP_label.config(text="{:<8} :".format("0.75 AP"))
        self.mAP_label.config(text="{:<8} :".format("mAP"))
        self.sort_by_dropdown.config(state="disabled")
        
        for image in self.my_images:
            image.detections = None

    def select_detection_button_click(self):
        # open the file dialog
        self.detection_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Detection File",filetypes=(("CSV files","*.csv"),("all files","*.*")))
        # print("Detection file : ", len(self.detection_file))
        
        if len(self.detection_file) == 0:
            messagebox.showerror("Error","Please select a detection file")
            return
        
        if self.detection_file_selected_flag:
            self.refresh_detections()
        
        self.detection_df = pd.read_csv(self.detection_file)
        
        if self.detection_df.shape[1] != 10:
            messagebox.showerror("Error","The detection file should have 10 columns")
            return
        
        self.detection_file_selected_flag = True

        self.detection_df.columns = ["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"]
        self.detection_df["image_id"] = self.detection_df["image_id"].values.astype('str')
        
        self.set_image_detections()
        self.show_detection_button.config(state="normal")
        if self.ground_truth_file is not None:
            self.calculate_metrics_button.config(state="normal")

        # set filter button state to normal
        self.car_button.config(state="normal")
        self.bus_button.config(state="normal")
        self.truck_button.config(state="normal")
        self.pedestrian_button.config(state="normal")
        self.all_button.config(state="normal")
        self.animal_button.config(state="normal")
        self.others_button.config(state="normal")

    def show_ground_truth_button_click(self):
        # print("Show detections flag : ",self.show_detections_flag)
        # print("Show ground truths flag : ",self.show_ground_truths_flag)

        ret = 0
        if self.show_ground_truths_flag is False:
            self.show_ground_truth_button.config(relief="sunken")
            if self.show_detections_flag is False:
                ret = self.show_ground_truths(self.current_image)
            else:
                ret = self.show_detections_and_ground_truths(self.current_image)
        else:
            self.show_ground_truth_button.config(relief="raised")
            if self.show_detections_flag is False:
                ret = self.show_original_image(self.current_image)
            else:
                ret = self.show_detections(self.current_image)
        if ret == 0:
            self.show_ground_truths_flag = not self.show_ground_truths_flag
        

    def show_detection_button_click(self):
        # print("Show detections flag : ",self.show_detections_flag)
        # print("Show ground truths flag : ",self.show_ground_truths_flag)
        
        ret = 0
        
        if self.show_detections_flag is False:
            self.show_detection_button.config(relief="sunken")
            if self.show_ground_truths_flag is False:
                ret = self.show_detections(self.current_image)
            else:
                ret = self.show_detections_and_ground_truths(self.current_image)
        else:
            self.show_detection_button.config(relief="raised")
            if self.show_ground_truths_flag is False:
                ret = self.show_original_image(self.current_image)
            else:
                ret = self.show_ground_truths(self.current_image)

        if ret == 0:
            self.show_detections_flag = not self.show_detections_flag
        


    def update_image_list(self):
        print("Updating Image List")

        self.image_names = os.listdir(self.current_folder)

        # print(self.image_names)
        # clear the listbox
        self.imageList.delete(0, tk.END)
        # add the files to the listbox if they are images
        for image_name in self.image_names:
            if image_name.endswith(".jpg") or image_name.endswith(".png"): 
                # print(self.image_dict[self.name_to_id[image_name]].toString())
                self.imageList.insert(tk.END, self.image_dict[self.name_to_id[image_name]].toString())

    def on_image_select(self, event):
        # get the selected image
        image_id = self.imageList.get(self.imageList.curselection()).split(" ")[0]

        self.current_image = self.image_dict[image_id].image_name

        # image should be stay selected on the listbox
        self.imageList.selection_set(self.imageList.curselection())

        # load the image to the canvas
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + self.current_image))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        # update the canvas
        self.canvas.update()

        self.refresh_show_buttons()

        # print(self.current_image)


    def picChangeHandler(self, event):
        """
        Runs when the selected picture is changed.
        Removes all bboxes from image and places new bboxes for the new image.
        Resets all zoom efects.
        """
        selection = event.widget.curselection()
        if selection:
            # After Picture is changed, we reset label list and selected label
            #self.resetdetections()
            self.picList.itemconfig(self.picIndex, bg='white', fg='black')
            
            # Find the image name of selected image and place new image on canvas
            self.picIndex = selection[0]
            imageFileName = event.widget.get(self.picIndex).split(':')[1]
            self.currentImageID = imageFileName.split('.')[0]

            self.currentImageBitmap = Image.open(self.currentFolder + "/" + imageFileName)
            self.selectedLabel = None
            self.canvasImageSrc = ImageTk.PhotoImage(self.currentImageBitmap)
            self.canvas.itemconfig(self.canvasImage, image=self.canvasImageSrc)
            self.picList.itemconfig(self.picIndex, bg='green', fg='white')

            # Reset scaling on canvas
            self.canvas.scale("all", 0, 0, 1.1**(self.scale), 1.1**(self.scale))
            self.scale = 0

            # Reset x and y views (scrolls)
            self.canvas.xview_moveto(self.origX)
            self.canvas.yview_moveto(self.origY)


    def loadJSONCatagories(self):
        try:
            with open('./catagories.json') as f:
                self.categories = json.load(f)
        except:
            print("Error: categories.json not found")

    def loadCSVdetections(fileName, detections, createFile = True):
        detections.clear();
        if(os.path.exists(fileName)):
            with open(fileName) as f:
                for l in f.readlines():
                    newLine = l.split(",");
                    picName = newLine[0].strip()
                    labelEntry = {
                        "picName": picName,
                        "id": int(newLine[1].strip()),
                        "xTop": round(max(0, float(newLine[2].strip())), 0),
                        "yTop": round(max(0, float(newLine[3].strip())), 0),
                        "xBot": round(float(newLine[4].strip()), 0),
                        "yBot": round(float(newLine[5].strip()), 0),
                        "xCenter": round(float(newLine[6].strip()), 0),
                        "yCenter": round(float(newLine[7].strip()), 0),
                        "label": newLine[8].strip(),
                        "score": round(float(newLine[9].strip()), 3) if len(newLine)>=10 else 0}
                    if picName in detections:
                        detections[picName].append(labelEntry)
                    else:
                        detections[picName] = [labelEntry]
        else:
            if createFile:
                try:
                    with open(fileName, 'w'):
                        pass
                except Exception as e:
                    print(e)

    def show_detections(self,image_name):
        # load the image to the canvas
        if image_name == None:
            messagebox.showerror("Error", "No image selected")
            self.show_detection_button.config(relief="raised")
            return 1
        self.detection_df['image_id'] = self.detection_df['image_id'].astype(str)
        labelArray = self.image_dict[self.name_to_id[image_name]].detections
        # print(" labelArray: ")
        # print(labelArray)

        # if labelArray is empty, show error message
        if labelArray.empty:
            messagebox.showerror("Error", "No object detected in the image")
            self.show_detection_button.config(relief="raised")
            return 1
        
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + image_name))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        # update the canvas
        # print("image_name: ", image_name)
        image_name = image_name.split('.')[0]
        # set picName as string
        # print(self.detection_df)
        # "image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"


        for label in labelArray.itertuples():
                # print("Label:",label)

                dummy,picName, id, xT, yT, xB, yB, xC, yC, label, score = label 

                
                xT = xT * (1.1**self.scale)
                yT = yT * (1.1**self.scale)
                xB = xB * (1.1**self.scale)
                yB = yB * (1.1**self.scale)
                labelRect = self.canvas.create_rectangle(xT, yT, xB, yB, outline='red', tags="rect", width=1)
                # add text to the canvas
                # find name of the label from the catJSON
                label_name = ""
                for j in range(len(self.categories)):

                    if self.categories[j]['id'] == label:
                        label_name = self.categories[j]['name']
                        break
                
                # print("label_name: ", label_name)
                label_txt = label_name + " " + str(np.around(score,decimals=2))
                self.canvas.create_text(xT, yT, text=label_txt, anchor="nw", tags="text",font=("Helvetica", 8))
                # add confidence score to the canvas

        # update the canvas
        self.canvas.update()

        return 0

    def show_ground_truths(self, image_name):
        if image_name == None:
            messagebox.showerror("Error", "No image selected")
            self.show_detection_button.config(relief="raised")
            return 1
        self.ground_truth_df['image_id'] = self.ground_truth_df['image_id'].astype(str)
        labelArray = self.image_dict[self.name_to_id[image_name]].ground_truths
        # print(" labelArray: ")
        # print(labelArray)

        # if labelArray is empty, show error message
        if labelArray.empty:
            messagebox.showerror("Error", "No object detected in the image")
            self.show_ground_truth_button.config(relief="raised")
            return 1
        # load the image to the canvas
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + image_name))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        # update the canvas
        # print("image_name: ", image_name)
        image_name = image_name.split('.')[0]
        # set picName as string
        # print(self.detection_df)
        # "image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"

        

        for label in labelArray.itertuples():
                # print("Label:",label)

                dummy,picName, id, xT, yT, xB, yB, xC, yC, label = label 

                if label == -1 or label == 1001: 
                    continue
                
                xT = xT * (1.1**self.scale)
                yT = yT * (1.1**self.scale)
                xB = xB * (1.1**self.scale)
                yB = yB * (1.1**self.scale)
                labelRect = self.canvas.create_rectangle(xT, yT, xB, yB, outline='green', tags="rect", width=1)
                # add text to the canvas
                # find name of the label from the catJSON
                label_name = ""
                for j in range(len(self.categories)):

                    if self.categories[j]['id'] == label:
                        label_name = self.categories[j]['name']
                        break
                
                # print("label_name: ", label_name)
                label_txt = label_name
                # put label text to right bottom corner of the rectangle
                self.canvas.create_text(xB, yB, text=label_txt, anchor="se", tags="text",font=("Helvetica", 8))
                # add confidence score to the canvas

        # update the canvas
        self.canvas.update()

        return 0

    def show_detections_and_ground_truths(self, image_name):
        if image_name == None:  
            messagebox.showerror("Error", "No image selected")
            self.show_detection_button.config(relief="raised")
            return 1 
        
        detectionsArray = self.image_dict[self.name_to_id[image_name]].detections
        # if detectionsArray is empty, show error message
        if detectionsArray.empty:
            messagebox.showerror("Error", "No object detected in the image")
            self.show_detection_button.config(relief="raised")
            return 1
        
        gtArray = self.image_dict[self.name_to_id[image_name]].ground_truths
        # if gtArray is empty, show error message
        if gtArray.empty:
            messagebox.showerror("Error", "No object detected in the image")
            self.show_ground_truth_button.config(relief="raised")
            return 1
        
        # load the image to the canvas
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + image_name))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img


    
        for det in detectionsArray.itertuples():
                # print("det:",det)

                dummy,picName, id, xT, yT, xB, yB, xC, yC, det, score = det 
                
                xT = xT * (1.1**self.scale)
                yT = yT * (1.1**self.scale)
                xB = xB * (1.1**self.scale)
                yB = yB * (1.1**self.scale)
                detRect = self.canvas.create_rectangle(xT, yT, xB, yB, outline='red', tags="rect", width=1)
                # add text to the canvas
                # find name of the det from the catJSON
                det_name = ""
                for j in range(len(self.categories)):

                    if self.categories[j]['id'] == det:
                        det_name = self.categories[j]['name']
                        break
                
                # print("det_name: ", det_name)
                det_txt = det_name + " " + str(np.around(score,decimals=2))
                self.canvas.create_text(xT, yT, text=det_txt, anchor="nw", tags="text",font=("Helvetica", 8))
                # add confidence score to the canvas

 
        for gt in gtArray.itertuples():

            dummy,picName, id, xT, yT, xB, yB, xC, yC, gt = gt 

            if gt == -1 or gt == 1001: 
                continue
            
            xT = xT * (1.1**self.scale)
            yT = yT * (1.1**self.scale)
            xB = xB * (1.1**self.scale)
            yB = yB * (1.1**self.scale)
            gtRect = self.canvas.create_rectangle(xT, yT, xB, yB, outline='green', tags="rect", width=1)
            # add text to the canvas
            # find name of the gt from the catJSON
            gt_name = ""
            for j in range(len(self.categories)):

                if self.categories[j]['id'] == gt:
                    gt_name = self.categories[j]['name']
                    break
            
            # print("gt_name: ", gt_name)
            gt_txt = gt_name
            # put gt text to right bottom corner of the rectangle
            self.canvas.create_text(xB, yB, text=gt_txt, anchor="se", tags="text",font=("Helvetica", 8))

        # update the canvas
        self.canvas.update()

        return 0


    def show_original_image(self, image_name):
        # show the original image
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + image_name))
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        self.canvas.update()
        return 0


    def refresh_show_buttons(self):
        # update the show buttons
        print("Refreshing show buttons")
        if self.show_detections_flag:
            self.show_detection_button.config(relief="raised")
            self.show_detections_flag = False
        if self.show_ground_truths_flag:
            self.show_ground_truth_button.config(relief="raised")
            self.show_ground_truths_flag = False