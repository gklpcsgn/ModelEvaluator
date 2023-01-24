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

coco_res = calc_coco_map(results)

import json
try:
    with open('../catagories.json') as f:
        categories = json.load(f)
except:
    print("Error: categories.json not found")

# we will create a dataframe for threshold 0.5
result_df = pd.DataFrame(results[0.5]).T

for i in result_df.index:

    for j in range(len(categories)):
                if categories[j]['id'] == i:
                    name = categories[j]['name']
                    break
    result_df.loc[i,"class"] = name

result_df.set_index('class',inplace=True)
result_df.index.name = None

map_df = pd.DataFrame([coco_res[0.5],coco_res[0.75],coco_res["mAP"]],columns=["AP"],index=[0.5,0.75,"mAP"])


