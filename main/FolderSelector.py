import json
import tkinter as tk
from tkinter import RAISED, SUNKEN, ttk
from tkinter import Frame, Toplevel, StringVar, Button, Event, Label, PhotoImage, Scrollbar, filedialog, image_names, simpledialog,Entry,W
import os
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from modelRunner import ModelRunner
import pandas as pd




# TODO! REMOVE OUTPUT FOLDERS AFTER RUNNING ON IMAGES






class FolderSelector:

    def __init__(self,root):
        self.root = root
        self.root.title("TyX Technologies Model Evaluation Tool")
        self.root.geometry("1280x800")
        self.current_folder = None
        self.current_image = None
        self.scale = 0
        self.current_model = None
        self.output_path = None
        self.modelRunner = ModelRunner(self)
        self.labels = pd.DataFrame(columns=['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score'])
        self.catagories = None
        self.initUI()

    def initUI(self):
        self.choose_image_button = Button(self.root, text="Choose Folder", command=self.select_image_button_click)
        self.choose_image_button.grid(row=0, column=0,sticky="n",padx=10,pady=10)

        # add a frame for list and scrollbar
        self.image_list_frame = Frame(self.root)
        self.image_list_frame.grid(row=1, column=0, sticky="n", padx=10,pady=10)

        self.imageList = tk.Listbox(self.image_list_frame, width=40, height=20)
        self.imageList.grid(row=0, column=0, sticky="n", padx=10,pady=10)
        self.imageList.bind('<<ListboxSelect>>', self.on_image_select)
        # add a scrollbar to the listbox
        self.scrollbar = Scrollbar(self.image_list_frame, orient="vertical")
        self.scrollbar.config(command=self.imageList.yview)
        self.scrollbar.grid(row=0, column=1, sticky="nsew")
        self.imageList.config(yscrollcommand=self.scrollbar.set)

        self.loadJSONCatagories()

        # print(self.categories)

        # create a listbox that contains model names and a title

        tk.Label(self.root, text="Select Model").grid(row=2, column=0, sticky="n", padx=10,pady=10)

        self.modelList = tk.Listbox(self.root, width=40, height=20, exportselection=False)
        self.modelList.grid(row=3, column=0, sticky="n", padx=10,pady=10)
        self.modelList.insert(1, "detectron2")
        self.modelList.insert(2, "detr")
        self.modelList.insert(3, "prbnet")
        self.modelList.insert(4, "yolov7")
        self.modelList.insert(5, "MaskDino - ade20k")
        self.modelList.insert(6, "MaskDino - cityscapes")
        

        # when a model is selected, change the self.current_model variable
        self.modelList.bind('<<ListboxSelect>>', self.on_model_select)

        # add a frame that contains 5 buttons
        self.button_frame = Frame(self.root)
        self.button_frame.grid(row=4, column=0,sticky="n",padx=10,pady=10)

        # add a button that runs on the selected image
        self.run_button = Button(self.button_frame, text="Run on Image", command=self.on_run_image_button_click)
        self.run_button.grid(row=0, column=0,sticky="ew",padx=10,pady=10)

        # add a button that runs on the whole folder
        self.run_folder_button = Button(self.button_frame, text="Run on Folder", command=self.on_run_folder_button_click)
        self.run_folder_button.grid(row=0, column=1,sticky="ew",padx=10,pady=10)

        self.image_frame = Frame(self.root,width=896,height=512,padx=10,pady=10)
        self.image_frame.grid(row=0, column=1,rowspan=4)

        self.canvas = tk.Canvas(self.image_frame,width=896,height=512)
        self.canvas.grid(row=0, column=0)

        self.scroll_x = tk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")

        self.scroll_y = tk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.below_image_frame = Frame(self.root)
        self.below_image_frame.grid(row=4, column=1,sticky="n",padx=10,pady=10)
        
        self.show_predictions_button = Button(self.below_image_frame, text="Show Predictions", command=self.on_show_predictions_button_click)
        self.show_predictions_button.grid(row=0, column=0,sticky="ew",padx=10,pady=10)

        self.show_original_button = Button(self.below_image_frame, text="Show Original Image", command=self.on_show_original_button_click)
        self.show_original_button.grid(row=0, column=1,sticky="ew",padx=10,pady=10)

        self.image_frame.grid_rowconfigure(0, weight=1) 
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(2, minsize=100)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.configure(yscrollincrement='1')
        self.canvas.configure(xscrollincrement='1')
        self.canvas.configure(xscrollcommand=self.scroll_x.set)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvasImage = self.canvas.create_image(0, 0, anchor="nw",image=None)

        # add a show prediction button
        self.select_label = Button(self.button_frame, text="Select Label",command=self.on_select_label_click)
        self.select_label.grid(row=1, column=0,sticky="ew",padx=10,pady=10)

        self.check_button_active()

    def on_select_label_click(self):
        # ask the user to select a csv file from the output folder
        # open the file dialog
        self.current_label_file = filedialog.askopenfilename(initialdir=os.getcwd() + "/main/output", title="Select Label File", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
        
        self.update_labels(self.current_label_file)

        self.show_predictions_button.config(relief=RAISED) 
        self.show_original_button.config(relief=RAISED)
        # open the csv file


    def update_labels(self, label_file):

        with open(label_file, 'r') as csv_file:
           # update self.labels
            self.labels = pd.read_csv(csv_file,header=None)
            self.labels.columns = ['picName', 'id', 'xTop', 'yTop', 'xBot', 'yBot','xCenter', 'yCenter', 'label', 'score']
            print(self.labels)
        csv_file.close()

    def on_show_predictions_button_click(self):
        self.show_predictions_button.config(relief=SUNKEN)
        self.show_original_button.config(relief=RAISED)
        # show the predictions
        self.show_labels(self.current_image)

    def on_show_original_button_click(self):
        self.show_predictions_button.config(relief=RAISED)  
        self.show_original_button.config(relief=SUNKEN)
        # show the original image
        self.show_original(self.current_image) 



    def after_segmentation(self, prediction):
        self.output_path = prediction
        self.after_run_on_image_segmentation()


    # select image button click event
    def select_image_button_click(self):
        # open the file dialog
        self.current_folder = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Folder")   

        self.modelRunner.inputfolder = self.current_folder
    

        # list all files in the folder
        self.image_names = os.listdir(self.current_folder)

        print(self.image_names)
        # clear the listbox
        self.imageList.delete(0, tk.END)
        # add the files to the listbox if they are images
        for image_name in self.image_names:
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                self.imageList.insert(tk.END, image_name)

        self.modelRunner.inputfolder = self.current_folder
        self.check_button_active()


        
    def on_model_select(self, event):
        # get the selected model
        self.current_model = self.modelList.get(self.modelList.curselection())
        self.modelRunner.modelName = self.current_model
        print(self.current_model)   
        self.check_button_active()

    def on_image_select(self, event):
        # get the selected image
        self.current_image = self.imageList.get(self.imageList.curselection())

        # image should be stay selected on the listbox
        self.imageList.selection_set(self.imageList.curselection())

        # load the image to the canvas
        self.modelRunner.imagePath = self.current_folder + "/" + self.current_image
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + self.current_image))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        # update the canvas
        self.canvas.update()

        print(self.current_image)
        self.check_button_active()

        self.show_predictions_button.config(relief=RAISED)  
        self.show_original_button.config(relief=SUNKEN)


    def on_run_image_button_click(self):
        self.modelRunner.run_on_image()
        # self.after_run_on_image_segmentation()
        self.check_button_active()
        self.show_predictions_button.config(relief=SUNKEN)
        self.show_original_button.config(relief=RAISED)


    def after_run_on_image_segmentation(self):
        # load the image to the canvas
        img = ImageTk.PhotoImage(Image.open(self.output_path))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        self.canvas.update()
        self.check_button_active()


    def on_run_folder_button_click(self):
        self.modelRunner.run_on_folder()
        self.check_button_active()


    def check_button_active(self):
        # if model is selected and image is selected, enable the run button
        if self.current_model is not None and self.current_image is not None:
            self.run_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.DISABLED)

        # if model is selected and folder is selected, enable the run folder button
        if self.current_model is not None and self.current_folder is not None:
            self.run_folder_button.config(state=tk.NORMAL)
        else:
            self.run_folder_button.config(state=tk.DISABLED)

        # if output folder is empty, disable the show prediction button
        files = os.listdir("main/output/")
        if len(files) == 0:
            self.select_label.config(state=tk.DISABLED)
        else:
            self.select_label.config(state=tk.NORMAL)

        # if canvas is empty, disable the show original button
        print(len(self.canvas.find_all()))
        if len(self.canvas.find_all()) == 1:
            self.show_original_button.config(state=tk.DISABLED)
            self.show_predictions_button.config(state=tk.DISABLED)
        else:
            self.show_original_button.config(state=tk.NORMAL)
            self.show_predictions_button.config(state=tk.NORMAL)

            
        
    
    def picChangeHandler(self, event):
        """
        Runs when the selected picture is changed.
        Removes all bboxes from image and places new bboxes for the new image.
        Resets all zoom efects.
        """
        selection = event.widget.curselection()
        if selection:
            # After Picture is changed, we reset label list and selected label
            #self.resetLabels()
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

    def loadCSVLabels(fileName, labels, createFile = True):
        labels.clear();
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
                    if picName in labels:
                        labels[picName].append(labelEntry)
                    else:
                        labels[picName] = [labelEntry]
        else:
            if createFile:
                try:
                    with open(fileName, 'w'):
                        pass
                except Exception as e:
                    print(e)

    def load_one_image_labels(self, labels):
        # add it to labels dataframe
        self.labels = labels
        print(self.labels)

        self.show_labels(self.current_image)

    def show_labels(self,image_name):
        # load the image to the canvas
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + image_name))
        # add image to the canvas
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        # update the canvas
        print("image_name: ", image_name)
        image_name = image_name.split('.')[0]
        # set picName as string
        self.labels['picName'] = self.labels['picName'].astype(str)
        labelArray = self.labels[self.labels['picName'] == image_name]
        print(" labelArray: ")
        print(labelArray)

        # if labelArray is empty, show error message
        if labelArray.empty:
            messagebox.showerror("Error", "No object detected in the image")
            return
        

        for label in labelArray.itertuples():
                print("Label:",label)

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
                
                print("label_name: ", label_name)
                label_txt = label_name + " " + str(np.around(score,decimals=2))
                self.canvas.create_text(xT, yT, text=label_txt, anchor="nw", tags="text")
                # add confidence score to the canvas

        # update the canvas
        self.canvas.update()

    def show_original(self, image_name):
        # show the original image
        img = ImageTk.PhotoImage(Image.open(self.current_folder + "/" + image_name))
        self.canvas.create_image(0,0, image=img, anchor="nw")
        self.canvas.image = img
        self.canvas.update()