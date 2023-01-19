import tkinter as tk
from tkinter import ttk
from tkinter import Frame, Toplevel, StringVar, Button, Event, Label, PhotoImage, Scrollbar, filedialog, image_names, simpledialog
import os
from tkinter.constants import BOTTOM, END, HORIZONTAL, RIGHT, VERTICAL, W, X, Y
from typing_extensions import Annotated
from PIL import Image, ImageTk
from functools import partial
import pandas as pd
import numpy as np
from functools import cmp_to_key
import pprint
import shutil

class ModelRunner(Toplevel):
    def __init__(self, root):
        self.root = root
        self.currentImageID = None
        self.currentFolder = None
        self.startX = None
        self.startY = None
        self.detectionsDict = {}
        self.scale = 0
        self.initElements()

    def initElements(self):
        pass

        