import numpy as np
import pandas as pd

def detection_df_to_boxes(outputs):
    '''
    Convert ground truth dataframe to dictionary of boxes
    Returns: dict of dicts of 'boxes' and 'scores'
    '''
    outputs.columns = ["image_id","object_id","x_top","y_top","x_bottom","y_bottom","x_center","y_center","class","confidence"]
    output_dict = {}
    temp_rows = pd.DataFrame(columns=outputs.columns)
    for index, row in outputs.iterrows():
        image_id = str(int(row['image_id']))
        # add image_id to temp_indexes until next image_id is not same as current image_id
        if(index + 1 < outputs.shape[0] and str(int(outputs.loc[index+1,'image_id'])) == image_id):
            rowdf = pd.Series(row).to_frame().T
            rowdf["image_id"] = rowdf["image_id"].astype(int)
            rowdf["class"] = rowdf["class"].astype(int)
            temp_rows = pd.concat([temp_rows, rowdf], axis=0)
            continue
        else:
            rowdf = pd.Series(row).to_frame().T
            rowdf["image_id"] = rowdf["image_id"].astype(int)
            rowdf["class"] = rowdf["class"].astype(int)
            temp_rows = pd.concat([temp_rows, rowdf], axis=0)
            # boxes = [[x_top,y_top,x_bottom,y_bottom],...]
            boxes = []
            scores = []
            for temp_index, temp_row in temp_rows.iterrows():
                boxes.append([temp_row['x_top'],temp_row['y_top'],temp_row['x_bottom'],temp_row['y_bottom']])
                scores.append(temp_row['confidence'])
            output_dict[image_id] = {"boxes":boxes,"scores":scores}
            temp_rows = pd.DataFrame()
    return output_dict


def gt_df_to_boxes(annot):
    '''
    Convert ground truth dataframe to dictionary of boxes
    Returns: dict of dicts of boxes
    '''
    output_dict = {}
    temp_rows = pd.DataFrame(columns=annot.columns)
    for index, row in annot.iterrows():
        image_id = str(int(row['image_id']))
    # add image_id to temp_indexes until next image_id is not same as current image_id
        if(index + 1 < annot.shape[0] and str(int(annot.loc[index+1,'image_id'])) == image_id):
            rowdf = pd.Series(row).to_frame().T
            rowdf["image_id"] = rowdf["image_id"].astype(int)
            rowdf["class"] = rowdf["class"].astype(int)
            temp_rows = pd.concat([temp_rows, rowdf], axis=0)
            continue
        else:
            rowdf = pd.Series(row).to_frame().T
            rowdf["image_id"] = rowdf["image_id"].astype(int)
            rowdf["class"] = rowdf["class"].astype(int)
            temp_rows = pd.concat([temp_rows, rowdf], axis=0)
        # boxes = [[x_top,y_top,x_bottom,y_bottom],...]
            boxes = []
            for temp_index, temp_row in temp_rows.iterrows():
                boxes.append([temp_row['x_top'],temp_row['y_top'],temp_row['x_bottom'],temp_row['y_bottom']])
            output_dict[image_id] = {"boxes":boxes}
            temp_rows = pd.DataFrame()
    return output_dict


def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou= calc_iou(gt_box, pred_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp= len(gt_match_idx)
        fp= len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

# if IoU ≥0.5, classify the object detection as True Positive(TP)
# if Iou <0.5, then it is a wrong detection and classify it as False Positive(FP)
# When a ground truth is present in the image and model failed to detect the object, classify it as False Negative(FN).
# True Negative (TN): TN is every part of the image where we did not predict an object. This metrics is not useful for object detection, hence we ignore TN.
# precision = TP / TP + FP
# recall = TP / TP + FN
# we will calculate precision and recall for each class
# we will calculate mAP for each class
# we will calculate mAP for all classes
def get_results(detections, ground_truths, iou_thr=0.5):
    # global TP,FP,FN,precision,recall
    TP = {}
    FP = {}
    FN = {}
    precision = {}
    recall = {}

    for i in detections.keys():
        TP[i] = [0]
        FP[i] = [0]
        FN[i] = [0]
        precision[i] = 0
        recall[i] = 0
    for i in detections.keys():
        # if ground truth keys does not contain the class,then all detections are false positives
        if i not in ground_truths.keys():
            FP[i] = [1] * detections[i].shape[0]
            continue

        # convert detections dataframe to dictionary of boxes and scores
        pred_bb = detection_df_to_boxes(detections[i])
        # convert ground truth dataframe to dictionary of boxes
        gt_boxes = gt_df_to_boxes(ground_truths[i])

        for image_id in pred_bb.keys():
            # print(image_id)
            # if image_id is not present in ground truth,then all detections are false positives
            if str(image_id) not in gt_boxes.keys():
                FP[i].append(len(pred_bb[image_id]["boxes"]))
                # print("skipped : ",image_id)
                continue
            
            results = get_single_image_results(gt_boxes[image_id]["boxes"],pred_bb[image_id]["boxes"],iou_thr)
            TP[i].append(results["true_positive"])
            FP[i].append(results["false_positive"])
            FN[i].append(results["false_negative"])

        TP[i] = sum(TP[i])
        FP[i] = sum(FP[i])
        FN[i] = sum(FN[i])
        precision[i] = TP[i] / (TP[i] + FP[i])  
        recall[i] = TP[i] / (TP[i] + FN[i])
            
        # print("class : ",i)
        # print("TP : ",TP[i])
        # print("FP : ",FP[i])
        # print("FN : ",FN[i])
        # print("precision : ",precision[i])
        # print("recall : ",recall[i])
    out_dict = {}
    for i in detections.keys():
        out_dict[i] = {"TP":TP[i],"FP":FP[i],"FN":FN[i],"precision":precision[i],"recall":recall[i]}
    return out_dict


def calc_avg(results):
    avg = 0
    for i in results.keys():
        # print(results[i]["precision"])
        avg += results[i]["precision"]
    return avg/len(results.keys())
    
# we will calculate coco mAP
# The primary challenge metric in COCO 2017 challenge is calculated as follows:

# AP is calculated for the IoU threshold of 0.5 for each class.
# Calculate the precision at every recall value(0 to 1 with a step size of 0.01), then it is repeated for IoU thresholds of 0.55,0.60,…,.95.
# Average is taken over all the 80 classes and all the 10 thresholds.
def calc_coco_map(results):
    mAP = 0
    for threshold in np.arange(0.5,1,0.05):
        threshold = round(threshold,2)
        avg = calc_avg(results[threshold])
        mAP +=avg
        print("mAP for threshold {} is {}".format(threshold,avg))
    print("mAP : " , mAP/10)
