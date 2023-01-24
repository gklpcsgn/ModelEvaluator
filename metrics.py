import numpy as np
import pandas as pd
from metric_functions import *
# from metrics import get_model_scores, calc_iou, calc_precision_recall, get_avg_precision_at_iou,get_single_image_results

# create a global dictionary of dataframes
global detections
detections = {}
global ground_truths
ground_truths = {}

det = pd.read_csv("/media/gklpcsgn/CE623CD9623CC84B/TYX/new-data/2021-07-25--11-00-PTZ-G13-01/output.csv",header=None)
det.columns = ["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"]

# create a dataframe for each class
for i in det["class"].unique():
    # add dataframe to global detections
    detections[i] = det[det["class"] == i].reset_index(drop=True)


annot = pd.read_csv("annot-test.csv")

for i in annot["class"].unique():
    # add dataframe to global ground_truths
    ground_truths[i] = annot[annot["class"] == i].reset_index(drop=True)


results = {}

for threshold in np.arange(0.5,1,0.05):
    threshold = round(threshold,2)
    results[threshold] = get_results(detections, ground_truths, iou_thr=threshold)

calc_coco_map(results)