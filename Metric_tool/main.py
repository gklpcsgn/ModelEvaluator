import tkinter as tk
from tkinter import ttk
from tkinter import Frame, Toplevel, StringVar, Button, Event, Label, PhotoImage, Scrollbar, filedialog, image_names, simpledialog,Entry,W
import os   
from PIL import Image, ImageTk
from FolderSelector import FolderSelector


global image


root = tk.Tk()

root.title("TyX Technologies Metric Calculation Tool")

folder_selector = FolderSelector(root)



root.mainloop()