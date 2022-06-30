import os

import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_optimizer(optimizer_name, parameters, learning_rate):
    """optimizer config

    Args:

    Returns:

    """
    if optimizer_name == "Adam":
        optimizer = Adam(parameters, learning_rate)

    return optimizer


def get_scheduler(optimizer, cfg):
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1
        )

    return scheduler


def get_img_list(data_dir, mode):
    """collect data paths"""
    file_list = []
    dir_path = os.path.join(data_dir, mode)
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_list.append(os.path.join(dirname, filename))

    file_list = sorted(file_list)

    return file_list


def save_image(_img, filename, mode):
    """
    Args:
        img: cpu(), [c,h,w]
    """
    if mode == "input":
        img = _img.clone() * 255.0
        img = img.clamp(0, 255)
        img = img.squeeze(0).numpy().astype("uint8")
        img = Image.fromarray(img, "L")
        img.save(filename)
        return img

    elif mode == "heatmap":
        img = _img.squeeze(0).numpy().astype("float32")
        tmp = np.uint8(cm.jet(img) * 255)
        heatmap = Image.fromarray(tmp)
        heatmap = np.asarray(heatmap.convert("RGB"))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, heatmap)

        return img

    elif mode == "heatmap_gray":
        img = _img.squeeze(0).numpy().astype("float32")
        tmp = np.uint8(img * 255)
        heatmap_gray = np.asarray(tmp)
        cv2.imwrite(filename, heatmap_gray)

    else:
        pass


def add_heatmap_to_orig(img, heatmap, save_name):
    """ """
    tmp = np.uint8(cm.jet(heatmap) * 255)
    _heatmap = Image.fromarray(tmp)
    _heatmap = np.asarray(_heatmap.convert("RGB"))
    img = np.asarray(img.convert("RGB"))
    blend_img = cv2.addWeighted(img, 0.5, _heatmap, 0.5, 0)
    cv2.imwrite(save_name, cv2.cvtColor(blend_img, cv2.COLOR_RGB2BGR))


def findclosestdistance(point, otherpoints):
    distances = [np.sqrt(np.sum((point - p) ** 2)) for p in otherpoints]
    return np.min(distances)


def compute_f1_score(
    img,
    ground_truth_points_coords,
    predicted_points_coords,
    closestpoint_relative_threshold=0.01,
    return_all_stats=False,
):
    closestpoint_THRESHOLD = img.shape[0] * closestpoint_relative_threshold
    if len(predicted_points_coords) > 600:
        print("Too many points predicted, score is 0")
        if return_all_stats:
            return 0, 0, 0
        return 0
    if len(predicted_points_coords) == 0:
        print("No points predicted, score is 0")
        if return_all_stats:
            return 0, 0, 0
        return 0
    if len(ground_truth_points_coords) == 0:
        print("No GT points here, score is 0.5")
        if return_all_stats:
            return 0.5, 0.5, 0.5
        return 0.5
    true_point_distances = [
        findclosestdistance(point, predicted_points_coords)
        for point in ground_truth_points_coords
    ]
    true_point_found = [x < closestpoint_THRESHOLD for x in true_point_distances]
    pred_point_distances = [
        findclosestdistance(point, ground_truth_points_coords)
        for point in predicted_points_coords
    ]
    pred_point_good = [x < closestpoint_THRESHOLD for x in pred_point_distances]
    tp = np.sum(true_point_found)
    fn = len(ground_truth_points_coords) - tp
    fp = len(predicted_points_coords) - np.sum(pred_point_good)
    if tp == 0:
        if return_all_stats:
            return 0, 0, 0
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if return_all_stats:
        return f1, precision, recall
    return f1
