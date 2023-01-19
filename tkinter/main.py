import tkinter as tk
from tkinter import ttk
from tkinter import Frame, Toplevel, StringVar, Button, Event, Label, PhotoImage, Scrollbar, filedialog, image_names, simpledialog,Entry,W
import os
from PIL import Image, ImageTk


global image


root = tk.Tk()

root.title("TyX Technologies Model Evaluation Tool")

'''
We will have 3 parts. Right will be for the image, left top will be for the image selection and left bottom will be for the model selection. We will use grid for this.
'''
# Left Frame
left_frame = Frame(root)
left_frame.grid(row=0, column=0, sticky="nsew")

# Right Frame
right_frame = Frame(root)
right_frame.grid(row=0, column=1, sticky="nsew")

# Left Top Frame
left_top_frame = Frame(left_frame)
left_top_frame.grid(row=0, column=0, sticky="nsew")

# Left Bottom Frame
left_bottom_frame = Frame(left_frame)
left_bottom_frame.grid(row=1, column=0, sticky="nsew")

# add a canvas in the right frame
canvas = tk.Canvas(right_frame)
canvas.grid(row=0, column=0, sticky="nswe")

# add a vertical scrollbar to the canvas
vsb = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
vsb.grid(row=0, column=1, sticky='ns')
canvas.configure(yscrollcommand=vsb.set)

# configure the canvas
canvas.configure(scrollregion=canvas.bbox("all"))

# create a frame in the canvas which will be scrolled with it
frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor='nw')

# add a horizontal scrollbar to the canvas
hsb = tk.Scrollbar(right_frame, orient="horizontal", command=canvas.xview)
hsb.grid(row=1, column=0, sticky='ew')
canvas.configure(xscrollcommand=hsb.set)

# add select model button to the left bottom frame
select_model_button = Button(left_bottom_frame, text="Select Model")
select_model_button.grid(row=0, column=0, sticky="nsew")

# add select image button to the left top frame
select_image_button = Button(left_top_frame, text="Select Image")
select_image_button.grid(row=0, column=0, sticky="nsew")

# show the image in the right frame using PIL
def show_image(image_path):
    img = ImageTk.PhotoImage(Image.open(image_path))
    # add image to the canvas
    canvas.create_image(0,0, image=img, anchor="nw")
    canvas.image = img
    # update the canvas
    canvas.update()

# select image button click event
def select_image_button_click(event):
    # open the file dialog
    image_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image", filetypes=(("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")))
    print(image_path)
    
    # show the image
    show_image(image_path)

# select model button click event
def select_model_button_click(event):
    # open the file dialog
    model_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Model", filetypes=(("Model Files", "*.h5"), ("All Files", "*.*")))
    print(model_path)

# select image button click event
select_image_button.bind("<Button-1>", select_image_button_click)

# select model button click event
select_model_button.bind("<Button-1>", select_model_button_click)








root.mainloop()