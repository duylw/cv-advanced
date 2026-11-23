"""Shared inference pipeline and utilities for CountZES evaluation scripts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import subprocess
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_GDINO_ROOT = os.path.join(_REPO_ROOT, "GroundingDINO")
if _GDINO_ROOT not in sys.path:
    sys.path.insert(0, _GDINO_ROOT)

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy, gaussian_kde
from skimage.feature import peak_local_max
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")

from CounTR import models_mae_cross
from CounTR.demo import load_image, run_one_image
from DSALVANet.utils.data_preprocess import preprocess
from DSALVANet.utils.fsol_modules import vis_DSALVANet
from DSALVANet.utils.model_helper import build_model
from DSALVANet.utils.PerSense_countr import vis_countr
from groundingdino.util.inference import Model
from per_segment_anything import SamPredictor, sam_model_registry
from resource_monitor import get_hardware_summary, monitor_resources


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_argument_parser(
    default_outdir: str,
    default_fsoc: str,
    default_countr_ckpt: str,
) -> argparse.ArgumentParser:
    """Create a unified CLI parser for all CountZES dataset entrypoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=default_outdir)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional SAM checkpoint path override.",
    )
    parser.add_argument(
        "--sam_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b", "vit_t"],
    )
    parser.add_argument("--ref_idx", type=str, default="00")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save density overlay visualizations under outputs/.",
    )
    parser.add_argument(
        "--fsoc",
        type=str,
        default=default_fsoc,
        choices=["countr", "DSALVANet"],
        help="Few-shot counter backend.",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="./DSALVANet/checkpoints/checkpoint_200.pth",
        help="Path to DSALVANet checkpoint.",
    )
    parser.add_argument(
        "--countr_ckpt",
        type=str,
        default=default_countr_ckpt,
        help="Path to CounTR checkpoint.",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.15,
        help="GroundingDINO box threshold (paper Sec. 4 = 0.15; original GitHub script = 0.20).",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.15,
        help="GroundingDINO text threshold (default = 0.15).",
    )
    parser.add_argument(
        "--sses_start_percentile",
        type=int,
        default=90,
        help="Initial percentile threshold in SSES (paper Sec. 3.1 = 90; GitHub script = 80).",
    )
    parser.add_argument(
        "--fix_sses_inversion",
        action="store_true",
        default=True,
        help="Paper-exact SSES scoring (no 1 - sim_clip inversion, default=True matching Eq. 2-3).",
    )
    parser.add_argument(
        "--github_sses_inversion",
        dest="fix_sses_inversion",
        action="store_false",
        help="Reproduce original GitHub repo behavior (enable sim_clip = 1 - sim_clip inside SSES).",
    )
    parser.add_argument(
        "--single_instance_mode",
        type=str,
        default="paper",
        choices=["paper", "unit", "le1"],
        help="Single-instance RoI count filter: 'paper' (1<c<2), 'unit' (0.5<=c<=1.5), 'le1' (0<c<=1.5).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional limit on number of images to evaluate (0 = run full dataset).",
    )
    parser.add_argument(
        "--stage_samples",
        type=int,
        default=5,
        help="Number of images for which to save 2x4 stage-by-stage PNGs in outputs/<outdir>/stage_samples/.",
    )
    return parser


def _ensure_rmsnorm() -> None:
    if not hasattr(nn, "RMSNorm"):
        class RMSNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-8):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(normalized_shape))

            def forward(self, x):
                norm_x = x.norm(2, dim=-1, keepdim=True)
                rms = norm_x * (1.0 / (x.size(-1) ** 0.5))
                return self.weight * x / (rms + self.eps)

        nn.RMSNorm = RMSNorm


def load_pipeline_models(args, log_dir: Optional[str] = None) -> Dict[str, object]:
    """Load SAM, GroundingDINO, counter, CLIP, and AnyUp models once."""
    with monitor_resources(
        "load_models",
        log_dir=log_dir,
        metadata={"fsoc": getattr(args, "fsoc", None), "sam_type": getattr(args, "sam_type", None)},
        verbose=bool(log_dir),
    ):
        return _load_pipeline_models_impl(args)


def _load_pipeline_models_impl(args) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("======> Load SAM")
    default_sam_ckpts = {
        "vit_h": "data/sam_vit_h_4b8939.pth",
        "vit_l": "data/sam_vit_l_0b3195.pth",
        "vit_b": "data/sam_vit_b_01ec64.pth",
        "vit_t": "weights/mobile_sam.pt",
    }
    sam_ckpt = args.ckpt if args.ckpt is not None else default_sam_ckpts[args.sam_type]
    sam = sam_model_registry[args.sam_type](checkpoint=sam_ckpt).to(device=device)
    sam.eval()
    for param in sam.parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)
    print("======> Done")

    print("======> Load Grounding Detector")
    gdino_cfg = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(
        model_config_path=gdino_cfg,
        model_checkpoint_path=gdino_ckpt,
    )
    print("======> Done")

    print("======> Load Object Counter")
    if args.fsoc == "countr":
        counter_model = models_mae_cross.__dict__["mae_vit_base_patch16"](
            norm_pix_loss="store_true"
        )
        counter_model.to(device)
        checkpoint = torch.load(args.countr_ckpt, map_location="cpu")
        counter_model.load_state_dict(checkpoint["model"], strict=False)
        counter_model.eval()
    elif args.fsoc == "DSALVANet":
        counter_model = build_model(args.weight, device)
    else:
        raise ValueError(f"Unsupported fsoc backend: {args.fsoc}")
    print("======> Done")

    print("======> Load CLIP")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("======> Done")

    _ensure_rmsnorm()
    anyup = torch.hub.load("wimmerth/anyup", "anyup", verbose=False, trust_repo=True).to(device).eval()

    return {
        "device": device,
        "sam": sam,
        "predictor": predictor,
        "grounding_dino": grounding_dino_model,
        "counter": counter_model,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "anyup": anyup,
    }


def _restore_sam_cache(predictor: SamPredictor, sam_cache: Tuple) -> None:
    features, input_size, original_size = sam_cache
    predictor.features = features
    predictor.input_size = input_size
    predictor.original_size = original_size
    predictor.is_image_set = True


def _run_counter(
    fsoc: str,
    counter_model,
    src_img_bgr: np.ndarray,
    img_path: str,
    ori_boxes_yxyx: List[List[int]],
    filename: str,
    device: torch.device,
):
    """Run CounTR or DSALVANet in inference mode without tracking gradients."""
    with torch.no_grad():
        if fsoc == "countr":
            samples, boxes, orig_boxes, pos, W, H, new_W, new_H = load_image(
                img_path, ori_boxes_yxyx
            )
            samples = samples.unsqueeze(0).to(device, non_blocking=True)
            boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
            result, _, density_pred, density_raw = run_one_image(
                samples,
                boxes,
                pos,
                counter_model,
                W,
                H,
                filename,
                new_W,
                new_H,
                device=device,
            )
            vis_output, count, density_pred = vis_countr(
                src_img_bgr, orig_boxes, result, filename, density_pred
            )
            return vis_output, count, density_pred, density_raw

        if fsoc == "DSALVANet":
            query, supports = preprocess(src_img_bgr, ori_boxes_yxyx, device)
            output = counter_model(query, supports)
            vis_output, count, density_pred, density_raw = vis_DSALVANet(
                src_img_bgr, ori_boxes_yxyx, output, filename
            )
            return vis_output, count, density_pred, density_raw

    raise ValueError(f"Unsupported fsoc backend: {fsoc}")


def _is_single_instance(count_val: float, mode: str) -> bool:
    if mode == "paper":
        return 1.0 < count_val < 2.0
    if mode == "unit":
        return 0.5 <= count_val <= 1.5
    if mode == "le1":
        return 0.0 < count_val <= 1.5
    raise ValueError(f"Unknown single_instance_mode: {mode}")


def _clamp_box_xyxy(box, W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = map(int, box)
    x0 = max(0, min(W - 1, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H, y1))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


def _compute_sam_sim_up(
    predictor: SamPredictor,
    sam_cache: Tuple,
    test_image_rgb: np.ndarray,
    box_xyxy,
) -> np.ndarray:
    H, W = test_image_rgb.shape[:2]
    x0, y0, x1, y1 = _clamp_box_xyxy(box_xyxy, W, H)
    exemplar = test_image_rgb[y0:y1, x0:x1]
    predictor.set_image(exemplar)
    ex_feat = predictor.features.squeeze(0)
    ex_emb = ex_feat.view(ex_feat.shape[0], -1).mean(1)
    ex_emb = ex_emb / (ex_emb.norm() + 1e-8)

    _restore_sam_cache(predictor, sam_cache)
    full_feat = predictor.features.squeeze(0)
    full_feat = full_feat / (full_feat.norm(dim=0, keepdim=True) + 1e-8)
    kernel = ex_emb.view(1, -1, 1, 1)
    sim_map = F.conv2d(full_feat.unsqueeze(0), kernel).squeeze(0).squeeze(0)
    sim_lowres = sim_map.unsqueeze(0).unsqueeze(0)

    sim_up = (
        predictor.model.postprocess_masks(
            sim_lowres,
            input_size=predictor.input_size,
            original_size=predictor.original_size,
        )
        .squeeze(0)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    return normalize_sim_map(sim_up)


def _box_count_from_density(box, density_map: np.ndarray) -> float:
    x0, y0, x1, y1 = map(int, box)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(density_map.shape[1] - 1, x1)
    y1 = min(density_map.shape[0] - 1, y1)
    if x1 < x0 or y1 < y0:
        return 0.0
    return float(density_map[y0 : y1 + 1, x0 : x1 + 1].sum())


def _to_imagenet_tensor(img_rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    assert img_rgb_uint8.ndim == 3 and img_rgb_uint8.shape[2] == 3
    t = torch.from_numpy(img_rgb_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t.to(device)


def run_countzes_on_image(
    image_path: str,
    filename: str,
    class_name: str,
    models: Dict[str, object],
    args,
    return_debug: bool = False,
):
    """Run the 3-stage CountZES pipeline (DAE -> DGE -> FCE) on one image."""
    device = models["device"]
    predictor: SamPredictor = models["predictor"]
    grounding_dino_model = models["grounding_dino"]
    counter_model = models["counter"]
    clip_model = models["clip_model"]
    clip_processor = models["clip_processor"]
    anyup = models["anyup"]

    src_img = cv2.imread(image_path)
    if src_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    test_image = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    H, W = test_image.shape[:2]

    # Cache full-image SAM features once per image
    predictor.set_image(test_image)
    sam_cache = (predictor.features, predictor.input_size, predictor.original_size)

    # ==================================================
    # Stage 1: Detection Anchored Exemplar (DAE)
    # ==================================================
    sim_clip = compute_clip_similarity_map(
        test_image, class_name, clip_model, clip_processor, device
    )
    sim_clip = normalize_sim_map(sim_clip)

    print(f"Target object: {[class_name]}")
    classes = [class_name]
    box_threshold = float(getattr(args, "box_threshold", 0.15))
    text_threshold = float(getattr(args, "text_threshold", 0.15))
    sses_start_pct = int(getattr(args, "sses_start_percentile", 90))
    detections = grounding_dino_model.predict_with_classes(
        image=src_img,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if detections is None or len(detections.xyxy) == 0:
        print("No detections found. Retrying with fallback class ['object']...")
        detections = grounding_dino_model.predict_with_classes(
            image=src_img,
            classes=["object"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    if detections is None or len(detections.xyxy) == 0:
        raise ValueError(f"No detections found for {filename} even after fallback.")

    entropies = [compute_clip_entropy(sim_clip, box) for box in detections.xyxy]
    confidences = detections.confidence.tolist()
    valid_indices = [i for i, e in enumerate(entropies) if np.isfinite(e)]
    if not valid_indices:
        raise ValueError("All boxes had invalid entropies.")

    valid_entropies = [entropies[i] for i in valid_indices]
    valid_confidences = [confidences[i] for i in valid_indices]
    norm_entropy = normalize(valid_entropies)
    alpha = 0.5
    scores = [
        alpha * c + (1 - alpha) * (1 - e)
        for c, e in zip(np.array(valid_confidences), norm_entropy)
    ]
    best_index = valid_indices[int(np.argmax(scores))]
    bbox_coord = detections.xyxy[best_index]
    initial_box = [int(v) for v in bbox_coord]

    invert_sim = not bool(getattr(args, "fix_sses_inversion", True))
    _restore_sam_cache(predictor, sam_cache)
    dae, dae_mask, _ = SSES(
        test_image,
        bbox_coord,
        predictor,
        clip_model,
        clip_processor,
        class_name,
        device,
        sim_clip=sim_clip,
        num_points=16,
        min_distance=5,
        start_percentile=sses_start_pct,
        min_percentile=50,
        step=10,
        filename=filename,
        image_cached=True,
        invert_sim=invert_sim,
    )
    if dae is not None:
        bbox_coord = dae

    sim_up = _compute_sam_sim_up(predictor, sam_cache, test_image, bbox_coord)

    if len(detections.xyxy) == 1:
        _restore_sam_cache(predictor, sam_cache)
        dae, dae_mask_refined, _ = SSES(
            test_image,
            bbox_coord,
            predictor,
            clip_model,
            clip_processor,
            class_name,
            device,
            sim_clip=sim_up,
            num_points=4,
            min_distance=5,
            start_percentile=90,
            min_percentile=80,
            step=10,
            filename=filename,
            image_cached=True,
            invert_sim=invert_sim,
        )
        if dae is not None:
            bbox_coord = dae
        if dae_mask_refined is not None:
            dae_mask = dae_mask_refined

    x0, y0, x1, y1 = _clamp_box_xyxy(bbox_coord, W, H)
    dae = [x0, y0, x1, y1]

    # ==================================================
    # Stage 2: Density Guided Exemplar (DGE)
    # ==================================================
    ori_boxes = [[y0, x0, y1, x1]]
    _, count_stage2, density_pred, density_raw = _run_counter(
        args.fsoc, counter_model, src_img, image_path, ori_boxes, filename, device
    )
    density_stage2 = density_pred.copy() if return_debug else None

    threshold = density_pred.mean() + 2 * density_pred.std()
    peaks = peak_local_max(density_pred, min_distance=5, threshold_abs=threshold)
    point_prompts = filter_peak_prompts(peaks, detections)

    _restore_sam_cache(predictor, sam_cache)
    _, box_list_sam = get_masks_and_boxes(
        predictor, test_image, point_prompts, image_cached=True
    )
    box_list_det = [
        [int(bx0), int(by0), int(bx1), int(by1)]
        for (bx0, by0, bx1, by1) in detections.xyxy
    ]
    box_list = box_list_sam + box_list_det + [dae]

    counts_per_box = [_box_count_from_density(b, density_raw) for b in box_list]
    single_instance_boxes1 = [
        b
        for b, c in zip(box_list, counts_per_box)
        if _is_single_instance(c, args.single_instance_mode)
    ]
    if not single_instance_boxes1:
        print("[Warning] No single-instance boxes found. Falling back to all boxes.")
        single_instance_boxes1 = box_list

    # Sequential execution on GPU avoids thread contention and keeps inference safe
    sam_box_count = [
        process_box(
            (det_box, image_path, counter_model, args, src_img, filename, device)
        )
        for det_box in single_instance_boxes1
    ]
    pseudo_gt_result = cluster_counts(sam_box_count, single_instance_boxes1, filename)
    pseudo_gt_count = pseudo_gt_result["kde_pseudo_gt"]

    entropies_all = [compute_clip_entropy(sim_up, box) for box in single_instance_boxes1]
    entropies_all = [e if np.isfinite(e) else 1.0 for e in entropies_all]
    norm_entropy = normalize(entropies_all)
    diffs = [abs(c - pseudo_gt_count) for c in sam_box_count]
    dmin, dmax = min(diffs), max(diffs)
    if dmax == dmin:
        closeness_norm = [1.0] * len(diffs)
    else:
        closeness_norm = [1.0 - ((d - dmin) / (dmax - dmin)) for d in diffs]

    scores = [
        alpha * c_close + (1 - alpha) * (1 - e)
        for c_close, e in zip(closeness_norm, norm_entropy)
    ]
    dge = single_instance_boxes1[int(np.argmax(scores))]

    # ==================================================
    # Stage 3: Feature Consensus Exemplar (FCE)
    # ==================================================
    x0, y0, x1, y1 = _clamp_box_xyxy(dge, W, H)
    dge = [x0, y0, x1, y1]
    dge_box = [[y0, x0, y1, x1]]

    sim_up_dge = _compute_sam_sim_up(predictor, sam_cache, test_image, dge)
    _, count_stage3, density_pred, density_raw = _run_counter(
        args.fsoc, counter_model, src_img, image_path, dge_box, filename, device
    )
    density_stage3 = density_pred.copy() if return_debug else None

    threshold = density_pred.mean() + 2 * density_pred.std()
    peaks2 = peak_local_max(density_pred, min_distance=5, threshold_abs=threshold)
    point_prompts2 = filter_peak_prompts(peaks2, detections)

    _restore_sam_cache(predictor, sam_cache)
    _, box_list_sam_new = get_masks_and_boxes(
        predictor, test_image, point_prompts2, image_cached=True
    )
    box_list_sam_new.append(dae)
    box_list_sam_new.append(dge)

    def _contains(box_a, box_b) -> bool:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        return ax0 <= bx0 and ay0 <= by0 and ax1 >= bx1 and ay1 >= by1

    filtered_boxes = []
    for i, box_a in enumerate(box_list_sam_new):
        if not any(i != j and _contains(box_a, box_b) for j, box_b in enumerate(box_list_sam_new)):
            filtered_boxes.append(box_a)
    if not filtered_boxes:
        print("[Warning] All boxes were filtered as containers. Falling back to original list.")
        filtered_boxes = box_list_sam_new

    counts_per_box = [_box_count_from_density(b, density_raw) for b in filtered_boxes]
    single_instance_boxes = [
        b
        for b, c in zip(filtered_boxes, counts_per_box)
        if _is_single_instance(c, args.single_instance_mode)
    ]
    if not single_instance_boxes:
        print("[Warning] No single-instance boxes found. Falling back to all boxes.")
        single_instance_boxes = filtered_boxes

    _restore_sam_cache(predictor, sam_cache)
    full_feat = predictor.features.squeeze(0)
    if not isinstance(full_feat, torch.Tensor):
        full_feat = torch.from_numpy(full_feat)
    C = full_feat.shape[0]
    hr_image = _to_imagenet_tensor(test_image, device)
    lr_features = full_feat.unsqueeze(0).to(device).contiguous()
    with torch.no_grad():
        hr_features = anyup(
            hr_image, lr_features, q_chunk_size=256, output_size=(256, 256)
        ).squeeze(0).contiguous()
    _, h_hr, w_hr = hr_features.shape

    def box_to_featvec(box_xyxy) -> torch.Tensor:
        bx0, by0, bx1, by1 = _clamp_box_xyxy(box_xyxy, W, H)
        fx0 = int(round(bx0 * (w_hr / float(W))))
        fx1 = int(round(bx1 * (w_hr / float(W))))
        fy0 = int(round(by0 * (h_hr / float(H))))
        fy1 = int(round(by1 * (h_hr / float(H))))
        fx0 = max(0, min(w_hr - 1, fx0))
        fx1 = max(fx0 + 1, min(w_hr, fx1))
        fy0 = max(0, min(h_hr - 1, fy0))
        fy1 = max(fy0 + 1, min(h_hr, fy1))

        region = hr_features[:, fy0:fy1, fx0:fx1]
        vec = region.reshape(C, -1).mean(dim=1)
        return vec / (vec.norm(p=2) + 1e-8)

    with torch.no_grad():
        vecs = torch.stack([box_to_featvec(b) for b in single_instance_boxes])
    if vecs.shape[0] == 1:
        fce = single_instance_boxes[0]
        kmeans_labels = [0]
    else:
        c0_idx = 0
        c1_idx = int(torch.argmin(vecs @ vecs[c0_idx]))
        centroids = torch.stack([vecs[c0_idx], vecs[c1_idx]])
        centroids = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)

        for _ in range(10):
            labels = torch.argmax(vecs @ centroids.T, dim=1)
            new_centroids = []
            for j in range(2):
                idx = (labels == j).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    mean_j = vecs[idx].mean(dim=0)
                    new_centroids.append(mean_j / (mean_j.norm(p=2) + 1e-8))
                else:
                    new_centroids.append(centroids[j])
            new_centroids = torch.stack(new_centroids, dim=0)
            if torch.allclose(new_centroids, centroids, atol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        sizes = torch.tensor([(labels == j).sum().item() for j in range(2)])
        majority = int(torch.argmax(sizes))
        maj_idx = (labels == majority).nonzero(as_tuple=False).squeeze(1)
        sims_to_centroid = vecs[maj_idx] @ centroids[majority]
        fce = single_instance_boxes[int(maj_idx[int(torch.argmax(sims_to_centroid))])]
        kmeans_labels = labels.detach().cpu().tolist()

    # Final counting with [dae, dge, fce]
    fce = [int(v) for v in fce]
    boxes = [dae, dge, fce]
    ori_boxes = [[int(b[1]), int(b[0]), int(b[3]), int(b[2])] for b in map(np.array, boxes)]
    vis_output, count, _, _ = _run_counter(
        args.fsoc, counter_model, src_img, image_path, ori_boxes, filename, device
    )

    debug_info = None
    if return_debug:
        debug_info = {
            "img_rgb": test_image,
            "sim_clip": sim_clip,
            "gdino_boxes": detections.xyxy.tolist(),
            "gdino_confs": detections.confidence.tolist(),
            "initial_box": initial_box,
            "dae_mask": dae_mask,
            "dae": dae,
            "sim_up_dae": sim_up,
            "density_stage2": density_stage2,
            "count_stage2": count_stage2,
            "point_prompts_stage2": point_prompts,
            "box_list_stage2": box_list,
            "single_instance_boxes1": single_instance_boxes1,
            "sam_box_count": sam_box_count,
            "pseudo_gt_count": pseudo_gt_count,
            "dge": [int(v) for v in dge],
            "sim_up_dge": sim_up_dge,
            "density_stage3": density_stage3,
            "count_stage3": count_stage3,
            "point_prompts_stage3": point_prompts2,
            "filtered_boxes": filtered_boxes,
            "single_instance_boxes": single_instance_boxes,
            "kmeans_labels": kmeans_labels,
            "fce": fce,
            "final_count": count,
        }

    del hr_features, lr_features, full_feat, hr_image, sam_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_debug:
        return vis_output, count, debug_info
    return vis_output, count


def load_class_map(image_class_path: str) -> Dict[str, str]:
    """Load tab-delimited filename -> class_name mapping once."""
    class_map = {}
    with open(image_class_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                class_map[parts[0]] = parts[1]
    return class_map


def get_class_name(filename: str, image_class_path: str) -> Optional[str]:
    with open(image_class_path, "r") as f:
        for line in f:
            file, class_name = line.strip().split("\t")
            if file == filename:
                return class_name
    return None


def _peaks_from_sim(
    sim_clip: np.ndarray,
    bbox_xyxy,
    num_points: int = 16,
    min_distance: int = 8,
    start_percentile: int = 80,
    min_percentile: int = 50,
    step: int = 10,
) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox_xyxy)
    H, W = sim_clip.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2), dtype=int)

    crop = sim_clip[y0 : y1 + 1, x0 : x1 + 1]
    crop_blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0, sigmaY=1.0)

    percentile = start_percentile
    pts = []
    while percentile >= min_percentile and len(pts) < num_points:
        thr = np.percentile(crop_blur, percentile)
        peaks = peak_local_max(
            crop_blur,
            min_distance=min_distance,
            threshold_abs=thr,
            exclude_border=False,
        )
        pts = [[x0 + int(px), y0 + int(py)] for (py, px) in peaks]
        if len(pts) >= num_points:
            break
        percentile -= step

    if len(pts) > num_points:
        pts = sorted(pts, key=lambda p: sim_clip[p[1], p[0]], reverse=True)[:num_points]

    return np.array(pts, dtype=int)


def compute_clip_entropy_masked(sim_map: np.ndarray, mask: np.ndarray, bins: int = 20) -> float:
    m = mask > 0
    if not np.any(m):
        return np.inf
    vals = sim_map[m].ravel()
    if np.all(vals == vals[0]):
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return np.inf
    hist = (hist + 1e-8) / (hist.sum() + 1e-8 * len(hist))
    return float(entropy(hist))


def SSES(
    test_image: np.ndarray,
    bbox_coord,
    predictor: SamPredictor,
    clip_model,
    clip_processor,
    class_text: str,
    device: torch.device,
    sim_clip: Optional[np.ndarray] = None,
    num_points: int = 16,
    min_distance: int = 5,
    start_percentile: int = 80,
    min_percentile: int = 50,
    step: int = 10,
    filename: Optional[str] = None,
    image_cached: bool = False,
    invert_sim: bool = True,
):
    """Similarity-guided SAM-based Exemplar Selection."""
    if sim_clip is None:
        raise ValueError("sim_clip must be provided.")

    if invert_sim:
        sim_clip = 1 - sim_clip

    H, W = test_image.shape[:2]
    x0, y0, x1, y1 = map(int, bbox_coord)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return bbox_coord, None, None

    global_sorted = np.sort(sim_clip.ravel())
    n_global = global_sorted.size
    points = _peaks_from_sim(
        sim_clip,
        [x0, y0, x1, y1],
        num_points=num_points,
        min_distance=min_distance,
        start_percentile=start_percentile,
        min_percentile=min_percentile,
        step=step,
    )
    if len(points) == 0:
        return bbox_coord, None, None

    if not image_cached or not predictor.is_image_set:
        predictor.set_image(test_image)

    candidates = []
    bins = 20
    h_max = np.log(bins)
    w_sim = 0.5
    w_ent = 0.5

    for pt in points:
        masks, _, _, _ = predictor.predict(
            point_coords=pt[None, :],
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if bx0 < x0 or bx1 > x1 or by0 < y0 or by1 > y1:
            continue

        fg = sim_clip[mask > 0]
        ent = compute_clip_entropy_masked(sim_clip, mask, bins=bins)
        if not np.isfinite(ent):
            continue
        ent_norm = float(np.clip(ent / (h_max + 1e-12), 0.0, 1.0))
        pct_ranks = np.searchsorted(global_sorted, fg, side="right") / n_global
        sim_score = float(pct_ranks.mean())
        comp = w_sim * sim_score + w_ent * (1.0 - ent_norm)
        candidates.append(
            {
                "score": comp,
                "mask": mask,
                "bbox": [bx0, by0, bx1, by1],
                "ent": float(ent_norm),
            }
        )

    coarse_mask = np.zeros(sim_clip.shape, dtype=np.uint8)
    coarse_mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    fg = sim_clip[coarse_mask > 0]
    if fg.size > 0:
        coarse_ent = compute_clip_entropy_masked(sim_clip, coarse_mask, bins=bins)
        if np.isfinite(coarse_ent):
            ent_norm = float(np.clip(coarse_ent / (h_max + 1e-12), 0.0, 1.0))
            pct_ranks = np.searchsorted(global_sorted, fg, side="right") / n_global
            sim_score = float(pct_ranks.mean())
            comp = w_sim * sim_score + w_ent * (1.0 - ent_norm)
            candidates.append(
                {
                    "score": comp,
                    "mask": coarse_mask.astype(bool),
                    "bbox": [x0, y0, x1, y1],
                    "ent": float(ent_norm),
                }
            )

    if not candidates:
        return bbox_coord, None, None
    best = max(candidates, key=lambda c: c["score"])
    return best["bbox"], best["mask"], best["ent"]


def cluster_counts(
    sam_box_count: List[int],
    box_list,
    filename: str,
    save_path: str = "./outputs/cluster_count",
) -> Dict[str, object]:
    if not sam_box_count:
        raise ValueError("Empty sam_box_count list.")

    counts_array = np.array(sam_box_count)
    std_dev = np.std(counts_array)
    try:
        if len(counts_array) > 1 and std_dev > 1e-6:
            kde = gaussian_kde(counts_array)
            xs = np.linspace(min(counts_array), max(counts_array), 500)
            kde_vals = kde(xs)
            kde_pseudo_gt = int(xs[np.argmax(kde_vals)])
        else:
            raise ValueError("Too few unique values for KDE")
    except Exception as e:
        print(f"[Warning] KDE failed for {filename} ({e}). Falling back to mean.")
        if len(set(counts_array)) == 1:
            kde_pseudo_gt = int(counts_array[0])
        else:
            kde_pseudo_gt = int(np.round(np.mean(counts_array)))

    lower_bound = kde_pseudo_gt - std_dev
    upper_bound = kde_pseudo_gt + std_dev
    kde_cluster_boxes = [
        i for i, c in enumerate(sam_box_count) if lower_bound <= c <= upper_bound
    ]
    return {
        "kde_pseudo_gt": kde_pseudo_gt,
        "kde_cluster_boxes": kde_cluster_boxes,
    }


def normalize_sim_map(sim_map: np.ndarray) -> np.ndarray:
    sim_min = sim_map.min()
    sim_max = sim_map.max()
    return (sim_map - sim_min) / (sim_max - sim_min + 1e-8)


def normalize(arr) -> np.ndarray:
    arr = np.array(arr)
    min_val, max_val = np.nanmin(arr), np.nanmax(arr)
    return (arr - min_val) / (max_val - min_val + 1e-8)


def process_box(box_args: Tuple) -> int:
    """Evaluate counter prediction for a single candidate exemplar box."""
    sam_box, img_path, counter_model, args, src_img, filename, device = box_args
    x0, y0, x1, y1 = map(int, sam_box)
    sam_box_yxyx = [[y0, x0, y1, x1]]

    try:
        _, count, _, _ = _run_counter(
            args.fsoc, counter_model, src_img, img_path, sam_box_yxyx, filename, device
        )
        return int(count)
    except Exception as e:
        print(f"Error processing box {sam_box_yxyx}: {e}")
        return -1


def compute_clip_entropy(clip_sim_map: np.ndarray, box, bins: int = 20) -> float:
    x0, y0, x1, y1 = map(int, box)
    H, W = clip_sim_map.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if y1 <= y0 or x1 <= x0:
        return np.inf
    region = clip_sim_map[y0:y1, x0:x1]
    if region.size == 0:
        return np.inf
    flat_region = region.flatten()
    if np.all(flat_region == flat_region[0]):
        return 0.0
    hist, _ = np.histogram(flat_region, bins=bins, range=(0, 1), density=True)
    hist += 1e-8
    hist /= hist.sum()
    return float(entropy(hist))


def get_masks_and_boxes(
    predictor: SamPredictor,
    image: np.ndarray,
    point_prompts,
    image_cached: bool = False,
):
    masks_list = []
    boxes_list = []
    if not image_cached or not predictor.is_image_set:
        predictor.set_image(image)

    for pt in point_prompts:
        masks, _, _, _ = predictor.predict(
            point_coords=np.array([pt]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]
        masks_list.append(mask)
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x0, y0 = np.min(x_indices), np.min(y_indices)
            x1, y1 = np.max(x_indices), np.max(y_indices)
            boxes_list.append([x0, y0, x1, y1])
        else:
            boxes_list.append([0, 0, 0, 0])
    return masks_list, boxes_list


def compute_clip_similarity_map(
    img_rgb: np.ndarray,
    class_name: str,
    clip_model,
    clip_processor,
    device: torch.device,
) -> np.ndarray:
    inputs = clip_processor(
        text=[class_name], images=[img_rgb], return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        patch_embeds = vision_outputs.last_hidden_state[:, 1:, :]
        text_outputs = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if hasattr(text_outputs, "pooler_output"):
            text_outputs = text_outputs.pooler_output
        patch_embeds = clip_model.visual_projection(patch_embeds)
    patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)
    text_emb = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
    sim = (patch_embeds @ text_emb.unsqueeze(-1)).squeeze(-1)
    side = int(math.sqrt(sim.shape[-1]))
    sim_map = sim.reshape(1, 1, side, side)
    sim_map = F.interpolate(
        sim_map,
        size=img_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return sim_map.detach().cpu().numpy()


def filter_peak_prompts(peak_points, detections):
    filtered_points = []
    boxes = np.array(detections.xyxy)
    for y, x in peak_points:
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes):
            filtered_points.append([x, y])
    return filtered_points


def update_and_log_metrics(count_item: int, mae: float, mse: float, pred_count, gt_count):
    print(f"GT_Count = {gt_count}")
    print(f"Pred_Count = {pred_count} \n")
    error = abs(float(pred_count) - float(gt_count))
    count_item += 1
    mae += error
    mse += error ** 2
    overall_mae = mae / count_item
    overall_rmse = (mse / count_item) ** 0.5
    print("MAE_Original %.2f, RMSE_Original %.2f \n" % (overall_mae, overall_rmse))
    return count_item, mae, mse


def _get_git_revision_info() -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = commit
        info["git_dirty"] = bool(dirty)
    except Exception:
        pass
    return info


def save_stage_diagnostic_grid(
    dbg: Dict[str, object],
    vis_output_bgr: np.ndarray,
    pred_count,
    gt_count,
    class_name: str,
    save_path: str,
) -> None:
    """Save an offscreen 2x4 stage-by-stage diagnostic grid PNG without mutating pyplot backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def _draw_box(img, box, color, label=None, thickness=2):
        if box is None or len(box) != 4:
            return
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        if label:
            cv2.putText(img, label, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    img_rgb = dbg["img_rgb"]

    vis_s1b = img_rgb.copy()
    if dbg["dae_mask"] is not None:
        mask_bool = dbg["dae_mask"].astype(bool)
        vis_s1b[mask_bool] = (0.65 * vis_s1b[mask_bool] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
    for b, c in zip(dbg["gdino_boxes"], dbg["gdino_confs"]):
        _draw_box(vis_s1b, b, (255, 220, 0), f"{c:.2f}", 1)
    _draw_box(vis_s1b, dbg["initial_box"], (255, 128, 0), "init", 2)
    _draw_box(vis_s1b, dbg["dae"], (0, 255, 0), "DAE", 3)

    vis_s2b = img_rgb.copy()
    for b in dbg["box_list_stage2"]:
        _draw_box(vis_s2b, b, (180, 180, 180), None, 1)
    for b in dbg["single_instance_boxes1"]:
        _draw_box(vis_s2b, b, (0, 220, 255), None, 2)
    _draw_box(vis_s2b, dbg["dge"], (0, 100, 255), "DGE", 3)

    vis_s3b = img_rgb.copy()
    for b, lbl in zip(dbg["single_instance_boxes"], dbg["kmeans_labels"]):
        col = (255, 165, 0) if lbl == 0 else (255, 0, 255)
        _draw_box(vis_s3b, b, col, f"c{lbl}", 2)
    _draw_box(vis_s3b, dbg["fce"], (255, 0, 0), "FCE", 3)

    fig = Figure(figsize=(24, 11))
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4)

    axes[0, 0].imshow(dbg["sim_clip"], cmap="jet")
    axes[0, 0].set_title(f"S1a. CLIP Sim Map ('{class_name}')")

    axes[0, 1].imshow(vis_s1b)
    axes[0, 1].set_title(f"S1b. GDINO ({len(dbg['gdino_boxes'])}) + SSES -> DAE (green)")

    axes[0, 2].imshow(dbg["sim_up_dae"], cmap="jet")
    axes[0, 2].set_title("S1c. SAM Sim Map (from DAE)")

    axes[0, 3].imshow(img_rgb)
    axes[0, 3].imshow(dbg["density_stage2"], cmap="jet", alpha=0.5)
    if len(dbg["point_prompts_stage2"]) > 0:
        pts2 = np.array(dbg["point_prompts_stage2"])
        axes[0, 3].scatter(pts2[:, 0], pts2[:, 1], c="white", marker="*", s=60, edgecolors="black")
    axes[0, 3].set_title(f"S2a. Density(DAE)={dbg['count_stage2']} + P2P ({len(dbg['point_prompts_stage2'])})")

    axes[1, 0].imshow(vis_s2b)
    axes[1, 0].set_title(f"S2b. GGES (KDE->{dbg['pseudo_gt_count']}) -> DGE (blue)")

    axes[1, 1].imshow(img_rgb)
    axes[1, 1].imshow(dbg["density_stage3"], cmap="jet", alpha=0.5)
    if len(dbg["point_prompts_stage3"]) > 0:
        pts3 = np.array(dbg["point_prompts_stage3"])
        axes[1, 1].scatter(pts3[:, 0], pts3[:, 1], c="white", marker="*", s=60, edgecolors="black")
    axes[1, 1].set_title(f"S3a. Density(DGE)={dbg['count_stage3']} + P2P ({len(dbg['point_prompts_stage3'])})")

    axes[1, 2].imshow(vis_s3b)
    axes[1, 2].set_title(f"S3b. FRES 2-Means ({len(dbg['single_instance_boxes'])}) -> FCE (red)")

    axes[1, 3].imshow(cv2.cvtColor(vis_output_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title(f"Final [DAE,DGE,FCE] -> Pred={pred_count} | GT={gt_count}")

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.print_png(save_path)
    fig.clf()


def run_recorded_evaluation(
    dataset_name: str,
    output_root: str,
    args,
    samples: List[Dict[str, object]],
    models: Dict[str, object],
) -> Dict[str, object]:
    """Run dataset evaluation while persisting config, per-image stage telemetry, resource logs, and summary."""
    from tqdm import tqdm

    max_images = int(getattr(args, "max_images", 0) or 0)
    if max_images > 0:
        samples = samples[:max_images]

    os.makedirs(output_root, exist_ok=True)
    log_dir = os.path.join(output_root, "logs")
    stage_dir = os.path.join(output_root, "stage_samples")
    jsonl_path = os.path.join(output_root, "predictions.jsonl")
    csv_path = os.path.join(output_root, "predictions.csv")
    config_path = os.path.join(output_root, "run_config.json")
    summary_path = os.path.join(output_root, "summary.json")

    run_config = {
        "dataset": dataset_name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "num_samples": len(samples),
        "args": {k: v for k, v in vars(args).items()},
        "git": _get_git_revision_info(),
        "hardware": get_hardware_summary(),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    stage_samples_limit = int(getattr(args, "stage_samples", 5) or 0)
    records: List[Dict[str, object]] = []
    failed_files: List[Dict[str, str]] = []
    count_item = 0
    mae = 0.0
    mse = 0.0
    total_gt = 0.0
    total_pred = 0.0
    stage_saved = 0
    eval_t0 = time.perf_counter()

    # Truncate previous predictions.jsonl for this output_root
    with open(jsonl_path, "w", encoding="utf-8"):
        pass

    print(f"======> Start Recorded Evaluation ({dataset_name}, {len(samples)} images)")
    print(f"Artifacts directory -> {output_root}")

    try:
        with monitor_resources(
            f"eval_{dataset_name.lower()}",
            log_dir=log_dir,
            metadata={
                "dataset": dataset_name,
                "num_samples": len(samples),
                "fsoc": getattr(args, "fsoc", None),
                "fix_sses_inversion": getattr(args, "fix_sses_inversion", False),
                "single_instance_mode": getattr(args, "single_instance_mode", "paper"),
            },
        ) as eval_mon:
            for idx, sample in enumerate(tqdm(samples, desc=dataset_name)):
                filename = str(sample["filename"])
                image_path = str(sample["image_path"])
                class_name = str(sample["class_name"])
                gt_count = float(sample["gt_count"])
                vis_save_path = sample.get("vis_save_path")

                img_t0 = time.perf_counter()
                row: Dict[str, object] = {
                    "index": idx,
                    "filename": filename,
                    "class_name": class_name,
                    "gt_count": int(gt_count) if gt_count.is_integer() else gt_count,
                    "pred_count": None,
                    "abs_error": None,
                    "sq_error": None,
                    "running_mae": None,
                    "running_rmse": None,
                    "duration_sec": None,
                    "status": "ok",
                    "gdino_boxes_n": None,
                    "initial_box": None,
                    "dae": None,
                    "count_stage2": None,
                    "p2p_stage2_n": None,
                    "sam_boxes_stage2_n": None,
                    "single1_n": None,
                    "pseudo_gt_kde": None,
                    "dge": None,
                    "count_stage3": None,
                    "p2p_stage3_n": None,
                    "filtered_n": None,
                    "single2_n": None,
                    "fce": None,
                }
                try:
                    vis_output, count, dbg = run_countzes_on_image(
                        image_path=image_path,
                        filename=filename,
                        class_name=class_name,
                        models=models,
                        args=args,
                        return_debug=True,
                    )
                    dur = round(time.perf_counter() - img_t0, 3)

                    if getattr(args, "visualize", False) and vis_save_path:
                        os.makedirs(os.path.dirname(str(vis_save_path)), exist_ok=True)
                        cv2.imwrite(str(vis_save_path), vis_output)

                    if stage_saved < stage_samples_limit:
                        stem = os.path.splitext(os.path.basename(filename))[0]
                        stage_png = os.path.join(stage_dir, f"{idx:04d}_{stem}_stages.png")
                        save_stage_diagnostic_grid(
                            dbg, vis_output, count, row["gt_count"], class_name, stage_png
                        )
                        stage_saved += 1

                    count_item, mae, mse = update_and_log_metrics(
                        count_item, mae, mse, count, gt_count
                    )
                    err = abs(float(count) - gt_count)
                    total_gt += gt_count
                    total_pred += float(count)

                    row.update(
                        {
                            "pred_count": count,
                            "abs_error": round(err, 4),
                            "sq_error": round(err ** 2, 4),
                            "running_mae": round(mae / count_item, 4),
                            "running_rmse": round((mse / count_item) ** 0.5, 4),
                            "duration_sec": dur,
                            "gdino_boxes_n": len(dbg["gdino_boxes"]),
                            "initial_box": dbg["initial_box"],
                            "dae": dbg["dae"],
                            "count_stage2": dbg["count_stage2"],
                            "p2p_stage2_n": len(dbg["point_prompts_stage2"]),
                            "sam_boxes_stage2_n": len(dbg["box_list_stage2"]),
                            "single1_n": len(dbg["single_instance_boxes1"]),
                            "pseudo_gt_kde": dbg["pseudo_gt_count"],
                            "dge": dbg["dge"],
                            "count_stage3": dbg["count_stage3"],
                            "p2p_stage3_n": len(dbg["point_prompts_stage3"]),
                            "filtered_n": len(dbg["filtered_boxes"]),
                            "single2_n": len(dbg["single_instance_boxes"]),
                            "fce": dbg["fce"],
                        }
                    )
                    del dbg, vis_output
                except Exception as exc:
                    dur = round(time.perf_counter() - img_t0, 3)
                    err_msg = f"error:{type(exc).__name__}:{exc}"
                    print(f"[ERROR] {filename}: {err_msg}")
                    row["duration_sec"] = dur
                    row["status"] = err_msg
                    failed_files.append({"filename": filename, "error": err_msg})

                records.append(row)
                with open(jsonl_path, "a", encoding="utf-8") as fj:
                    fj.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        total_dur = round(time.perf_counter() - eval_t0, 3)
        if records:
            fieldnames = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as fc:
                writer = csv.DictWriter(fc, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    writer.writerow(
                        {
                            k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                            for k, v in r.items()
                        }
                    )

        overall_mae = round(mae / count_item, 4) if count_item > 0 else None
        overall_rmse = round((mse / count_item) ** 0.5, 4) if count_item > 0 else None
        summary = {
            "dataset": dataset_name,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "fsoc": getattr(args, "fsoc", None),
            "sam_type": getattr(args, "sam_type", None),
            "fix_sses_inversion": getattr(args, "fix_sses_inversion", False),
            "single_instance_mode": getattr(args, "single_instance_mode", "paper"),
            "total_samples": len(samples),
            "succeeded": count_item,
            "failed": len(failed_files),
            "failed_files": failed_files,
            "MAE": overall_mae,
            "RMSE": overall_rmse,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_duration_sec": total_dur,
            "avg_sec_per_image": round(total_dur / len(records), 3) if records else None,
            "resource_eval": eval_mon.get("record") if "eval_mon" in locals() else None,
            "artifacts": {
                "run_config": config_path,
                "summary": summary_path,
                "predictions_csv": csv_path,
                "predictions_jsonl": jsonl_path,
                "resource_logs_dir": log_dir,
                "stage_samples_dir": stage_dir,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as fs:
            json.dump(summary, fs, indent=2, ensure_ascii=False)

        print("\n=== EVALUATION SUMMARY ===")
        print(
            f"Dataset={dataset_name} | Succeeded={count_item}/{len(samples)} | Failed={len(failed_files)} | "
            f"MAE={overall_mae} | RMSE={overall_rmse} | Time={total_dur}s"
        )
        print(f"Saved evidence -> {output_root}")

    return summary


def evaluate_mask_annotated_dataset(
    dataset_subdir: str,
    default_outdir: str,
    default_fsoc: str,
    default_countr_ckpt: str = "./CounTR/output_allnew_dir/FSC147.pth",
) -> None:
    """Shared evaluation loop for datasets using <id>.png and <id>A.png masks (MBM, VGG)."""
    parser = build_argument_parser(
        default_outdir=default_outdir,
        default_fsoc=default_fsoc,
        default_countr_ckpt=default_countr_ckpt,
    )
    args = parser.parse_args()
    print("Args:", args)

    os.makedirs("./outputs", exist_ok=True)
    output_path = os.path.join("./outputs", args.outdir)
    models = load_pipeline_models(args, log_dir=os.path.join(output_path, "logs"))

    images_path = os.path.join(args.data, dataset_subdir, "Images")
    samples: List[Dict[str, object]] = []
    for obj_name in sorted(os.listdir(images_path)):
        test_images_path = os.path.join(images_path, obj_name)
        if ".DS" in obj_name or not os.path.isdir(test_images_path):
            continue
        obj_output_path = os.path.join(output_path, obj_name)
        img_files = sorted(
            f
            for f in os.listdir(test_images_path)
            if f.endswith(".png") and not f.endswith("A.png")
        )
        for img_file in img_files:
            test_idx = os.path.splitext(img_file)[0]
            test_image_path = os.path.join(test_images_path, img_file)
            anno_image_path = os.path.join(test_images_path, f"{test_idx}A.png")
            anno_img = cv2.imread(anno_image_path, cv2.IMREAD_GRAYSCALE)
            if anno_img is None:
                raise FileNotFoundError(f"Annotation mask not found: {anno_image_path}")
            _, thresh = cv2.threshold(anno_img, 1, 255, cv2.THRESH_BINARY)
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(thresh)
            samples.append(
                {
                    "filename": img_file,
                    "image_path": test_image_path,
                    "class_name": obj_name,
                    "gt_count": int(num_labels - 1),
                    "vis_save_path": os.path.join(obj_output_path, f"{test_idx}.png"),
                }
            )

    run_recorded_evaluation(dataset_subdir.upper(), output_path, args, samples, models)
