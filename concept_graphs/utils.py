import json
import os
import torch
import numpy as np
import random
import logging
import re
import open3d as o3d
from pathlib import Path
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

# A logger for this file
log = logging.getLogger(__name__)


def load_point_cloud(path):
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))

    with open(path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)

    # Build a pcd with random colors
    pcd_o3d = []

    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d


def set_seed(seed: int = 42) -> None:
    # From wanb https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"Random seed set as {seed}")


def split_camel_preserve_acronyms(name):
    # Insert space between lowercase → uppercase
    # OR between acronym → normal word
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return s.lower()


def fetch_run_id(cfg):
    scene = OmegaConf.select(cfg, "dataset.scene")
    if scene:
        run_id = int(scene.split("_")[-1])
    else:
        run_id = 0
    return run_id


def aabb_iou(c1: np.ndarray, c2: np.ndarray) -> float:
    min1 = c1.min(axis=0)
    max1 = c1.max(axis=0)
    min2 = c2.min(axis=0)
    max2 = c2.max(axis=0)

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
    if inter_vol <= 0:
        return 0.0

    vol1 = float(np.prod(max1 - min1))
    vol2 = float(np.prod(max2 - min2))
    if vol1 <= 0 or vol2 <= 0:
        return 0.0

    return inter_vol / (vol1 + vol2 - inter_vol)


def procthor_to_ros(pose: dict) -> np.ndarray:
    """Convert Procthor pose (x, y, z, qw, qx, qy, qz) to ROS pose (x, y, z, qx, qy, qz, qw)."""
    LHS_TO_RHS = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    RHS_TO_ROS = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    CAM_OFFSET = 0.675
    position = pose['position']
    rotation = pose['rotation']

    r_yaw = R.from_euler("y", rotation["y"], degrees=True).as_matrix()
    r_pitch = R.from_euler("x", rotation["x"], degrees=True).as_matrix()

    # Combined rotation in Unity Frame
    rot_unity = r_yaw @ r_pitch
    pose_unity = np.eye(4)
    pose_unity[0:3, 0:3] = rot_unity
    pose_unity[0:3, 3] = [
        position["x"],
        position["y"] + CAM_OFFSET,
        position["z"],
    ]

    pose_rhs = RHS_TO_ROS @ (LHS_TO_RHS @ pose_unity @ LHS_TO_RHS.T)
    return pose_rhs
