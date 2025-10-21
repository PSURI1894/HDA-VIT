import numpy as np
from scipy.spatial.distance import cdist

def match_detections_to_gts(pred_centers, gt_centers, max_radius_mm=10.0, spacing=(1.0,1.0,1.0)):
    if len(pred_centers)==0:
        return [], list(range(len(gt_centers)))
    p_mm = np.array(pred_centers) * np.array(spacing)
    g_mm = np.array(gt_centers) * np.array(spacing)
    d = cdist(p_mm, g_mm)
    tp = set(); matched_gt=set()
    for pi in range(d.shape[0]):
        gj = np.argmin(d[pi])
        if d[pi, gj] <= max_radius_mm:
            if gj not in matched_gt:
                tp.add(pi)
                matched_gt.add(gj)
    FP = len(pred_centers) - len(tp)
    FN = len(gt_centers) - len(matched_gt)
    return { "TP":len(tp), "FP":FP, "FN":FN }

def compute_froc_at_thresholds(all_pred_scores, all_pred_centers, all_gt_centers, spacing=(1.0,1.0,1.0), thresholds=None, max_radius_mm=10.0):
    if thresholds is None:
        thresholds = np.linspace(0.99, 0.1, 20)
    sens = []
    mean_fp = []
    for th in thresholds:
        TP_total = 0; FN_total=0; FP_total=0; scans = len(all_pred_centers)
        for preds, scores, gts in zip(all_pred_centers, all_pred_scores, all_gt_centers):
            preds_filtered = [p for p,s in zip(preds, scores) if s >= th]
            res = match_detections_to_gts(preds_filtered, gts, max_radius_mm=max_radius_mm, spacing=spacing)
            TP_total += res['TP']; FP_total += res['FP']; FN_total += res['FN']
        sens.append(TP_total / (TP_total + FN_total + 1e-8))
        mean_fp.append(FP_total / scans)
    return thresholds, sens, mean_fp
