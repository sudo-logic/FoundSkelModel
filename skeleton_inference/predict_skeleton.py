import argparse
import contextlib
import glob
import io
import math
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import project modules from repository root.
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "data_gen"))

from model.DSTE import DST_Layer, Downstream, trunc_normal_


NTU_60_CLASSES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up",
    "throw", "sit down", "stand up", "clapping", "reading", "writing",
    "tear up paper", "put on jacket", "take off jacket", "put on a shoe",
    "take off a shoe", "put on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping", "jump up", "phone call",
    "play with phone/tablet", "type on a keyboard", "point to something", "taking a selfie",
    "check time (from watch)", "rub two hands", "nod head/bow", "shake head",
    "wipe face", "salute", "put palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)",
    "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person", "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other",
]


MAX_BODY_TRUE = 2


def class_name(idx: int) -> str:
    if 0 <= idx < len(NTU_60_CLASSES):
        return NTU_60_CLASSES[idx]
    return f"class_{idx}"


def resolve_cli_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    if os.path.exists(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(base_dir, value))


def parse_label_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path)
    if "A" not in name:
        return None
    try:
        return int(name.split("A", 1)[1][:3]) - 1
    except Exception:
        return None


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def read_skeleton_filter(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        skeleton_sequence: Dict = {"numFrame": int(f.readline()), "frameInfo": []}
        for _ in range(skeleton_sequence["numFrame"]):
            frame_info: Dict = {"numBody": int(f.readline()), "bodyInfo": []}
            for _ in range(frame_info["numBody"]):
                body_info_key = [
                    "bodyID", "clipedEdges", "handLeftConfidence",
                    "handLeftState", "handRightConfidence", "handRightState",
                    "isResticted", "leanX", "leanY", "trackingState",
                ]
                body_info = {
                    k: float(v) for k, v in zip(body_info_key, f.readline().split())
                }
                body_info["numJoint"] = int(f.readline())
                body_info["jointInfo"] = []
                for _ in range(body_info["numJoint"]):
                    joint_info_key = [
                        "x", "y", "z", "depthX", "depthY", "colorX", "colorY",
                        "orientationW", "orientationX", "orientationY",
                        "orientationZ", "trackingState",
                    ]
                    joint_info = {
                        k: float(v) for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info["jointInfo"].append(joint_info)
                frame_info["bodyInfo"].append(body_info)
            skeleton_sequence["frameInfo"].append(frame_info)
    return skeleton_sequence


def get_nonzero_std(s: np.ndarray) -> float:
    index = s.sum(-1).sum(-1) != 0
    s = s[index]
    if len(s) != 0:
        return float(s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std())
    return 0.0


def read_xyz(file_path: str, max_body: int = 4, num_joint: int = 25) -> np.ndarray:
    seq_info = read_skeleton_filter(file_path)
    data = np.zeros((max_body, seq_info["numFrame"], num_joint, 3))
    for n, frame in enumerate(seq_info["frameInfo"]):
        for m, body in enumerate(frame["bodyInfo"]):
            for j, v in enumerate(body["jointInfo"]):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v["x"], v["y"], v["z"]]
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:MAX_BODY_TRUE]
    data = data[index]
    return data.transpose(3, 1, 2, 0)  # (C, T, V, M)


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / math.sqrt(float(np.dot(axis, axis)))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0.0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def pre_normalization(
    data: np.ndarray,
    center_joint: int = 1,
    zaxis: List[int] = [11, 5],
    xaxis: List[int] = [],
) -> np.ndarray:
    # data: N, C, T, V, M
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, M, T, V, C

    # Pad null frames.
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = person.sum(-1).sum(-1) != 0
                tmp = person[index].copy()
                person *= 0
                person[: len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0 and person[i_f:].sum() == 0 and i_f > 0:
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    s[i_s, i_p, i_f:] = pad
                    break

    # Center skeleton.
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, center_joint : center_joint + 1, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    def align_human_to_vector(joint_idx1: int, joint_idx2: int, target_vector: List[float]) -> None:
        for i_s, skeleton in enumerate(s):
            if skeleton.sum() == 0:
                continue
            joint1 = skeleton[0, 0, joint_idx1]
            joint2 = skeleton[0, 0, joint_idx2]
            axis = np.cross(joint2 - joint1, target_vector)
            angle = angle_between(joint2 - joint1, np.array(target_vector))
            matrix = rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix, joint)

    if zaxis:
        align_human_to_vector(zaxis[0], zaxis[1], [0, 0, 1])
    if xaxis:
        align_human_to_vector(xaxis[0], xaxis[1], [1, 0, 0])

    return np.transpose(s, [0, 4, 2, 3, 1])  # N, C, T, V, M


def crop_subsequence(
    input_data: np.ndarray, num_of_frames: int, l_ratio: List[float], output_size: int
) -> np.ndarray:
    C, T, V, M = input_data.shape
    del T  # unused but kept for readability

    # Testing behavior from original feeder: center crop when l_ratio != 0.1
    start = int((1 - l_ratio[0]) * num_of_frames / 2)
    data = input_data[:, start : num_of_frames - start, :, :]
    temporal_crop_length = data.shape[1]

    temporal_crop = torch.tensor(data, dtype=torch.float32)
    temporal_crop = (
        temporal_crop.permute(0, 2, 3, 1)
        .contiguous()
        .view(C * V * M, temporal_crop_length)
    )
    temporal_crop = temporal_crop[None, :, :, None]
    temporal_crop = F.interpolate(
        temporal_crop, size=(output_size, 1), mode="bilinear", align_corners=False
    )
    temporal_crop = temporal_crop.squeeze(3).squeeze(0)
    temporal_crop = (
        temporal_crop.contiguous()
        .view(C, V, M, output_size)
        .permute(0, 3, 1, 2)
        .contiguous()
        .numpy()
    )
    return temporal_crop


def preprocess_skeleton_file(path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    data = read_xyz(path, max_body=2, num_joint=25)  # (C, T, V, M)
    num_frames = data.shape[1]
    # Silence verbose preprocess output for single-file inference.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        data = pre_normalization(data[None, ...])[0]
    raw = crop_subsequence(data, num_frames, [1.0], 64)
    jt = raw.transpose(1, 3, 2, 0).reshape(64, 2 * 25 * 3).astype("float32")
    js = raw.transpose(3, 2, 1, 0).reshape(2 * 25, 64 * 3).astype("float32")
    return torch.from_numpy(jt).unsqueeze(0), torch.from_numpy(js).unsqueeze(0), raw


def adapt_dste_spatial_layers(model: Downstream, state_dict: Dict[str, torch.Tensor]) -> None:
    key = "backbone.spe"
    if key not in state_dict:
        return

    target_len = int(state_dict[key].shape[1])
    current_len = int(model.backbone.spe.shape[1])
    if target_len == current_len:
        return

    hidden = int(model.backbone.spe.shape[2])
    num_head = int(model.backbone.t_tr.CA.attn.num_heads)
    alpha = float(model.backbone.t_tr.alpha)
    beta = float(model.backbone.t_tr.beta)
    gap = int(model.backbone.t_tr.DSA.gap)
    kernel = int(model.backbone.t_tr.CA.conv1.kernel_size[0])

    model.backbone.spe = nn.Parameter(torch.zeros(1, target_len, hidden))
    trunc_normal_(model.backbone.spe, std=0.02)
    attn0 = nn.MultiheadAttention(hidden, num_head, dropout=0.0, batch_first=True)
    attn1 = nn.MultiheadAttention(hidden, num_head, dropout=0.0, batch_first=True)
    model.backbone.s_tr = DST_Layer(target_len, hidden, alpha, beta, gap, attn0, kernel)
    model.backbone.s_tr1 = DST_Layer(target_len, hidden, alpha, beta, gap, attn1, kernel)
    print(f"Patched DSTE spatial sequence length: {current_len} -> {target_len}")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[Downstream, bool]:
    model = Downstream(
        t_input_size=150,
        s_input_size=192,
        hidden_size=1024,
        num_head=1,
        num_layer=2,
        num_class=60,
        modality="joint",
        alpha=0.5,
        gap=4,
        kernel_size=1,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = strip_module_prefix(raw)

    adapt_dste_spatial_layers(model, state)
    msg = model.load_state_dict(state, strict=False)
    has_classifier_head = "fc.weight" in state and "fc.bias" in state

    if has_classifier_head:
        print("Checkpoint mode: classifier")
    else:
        print("Checkpoint mode: feature-only (no fc head)")
    if msg.unexpected_keys:
        print(f"Warning: unexpected keys (showing first 5): {msg.unexpected_keys[:5]}")

    model.to(device)
    model.eval()
    return model, has_classifier_head


def infer_feature(model: Downstream, jt: torch.Tensor, js: torch.Tensor, device: torch.device) -> torch.Tensor:
    jt = jt.to(device, non_blocking=True)
    js = js.to(device, non_blocking=True)
    with torch.no_grad():
        feat = model(jt, js, None, None, None, None, knn_eval=True)
    return F.normalize(feat, p=2, dim=1).cpu()


def infer_classifier(model: Downstream, jt: torch.Tensor, js: torch.Tensor, device: torch.device) -> torch.Tensor:
    jt = jt.to(device, non_blocking=True)
    js = js.to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(jt, js, None, None, None, None, knn_eval=False)
    return logits.cpu()


def collect_skeleton_files(dataset_dir: str) -> List[str]:
    files = [
        os.path.join(dataset_dir, x)
        for x in os.listdir(dataset_dir)
        if x.endswith(".skeleton")
    ]
    files.sort()
    if not files:
        raise RuntimeError(f"No .skeleton files found in: {dataset_dir}")
    return files


def build_gallery(
    model: Downstream,
    dataset_dir: str,
    query_path: str,
    gallery_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_abs = os.path.abspath(query_path)
    files = [x for x in collect_skeleton_files(dataset_dir) if os.path.abspath(x) != query_abs]
    if not files:
        raise RuntimeError("Gallery is empty after excluding the query skeleton.")

    if gallery_size > 0 and len(files) > gallery_size:
        rnd = random.Random(seed)
        files = rnd.sample(files, gallery_size)

    feats = []
    labels = []
    for path in tqdm(files, desc="Building gallery"):
        gt = parse_label_from_filename(path)
        if gt is None:
            continue
        jt, js, _ = preprocess_skeleton_file(path)
        feats.append(infer_feature(model, jt, js, device)[0])
        labels.append(gt)

    if not feats:
        raise RuntimeError("No valid gallery samples with parseable labels.")

    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)


def print_metrics(
    skeleton_path: str,
    mode: str,
    pred: int,
    score: float,
    gt: Optional[int],
    topk: List[Tuple[int, float]],
) -> None:
    print("\n=== Recognition Result ===")
    print(f"skeleton_file: {skeleton_path}")
    print(f"mode: {mode}")
    if gt is not None:
        print(f"original_label: A{gt + 1:03d} ({class_name(gt)})")
    else:
        print("original_label: N/A (filename does not encode class id)")
    print(f"predicted_label: A{pred + 1:03d} ({class_name(pred)})")
    print(f"score: {score:.4f}")
    if gt is not None:
        print(f"correctness: {pred == gt}")
    else:
        print("correctness: N/A")
    print("topk:")
    for idx, val in topk:
        print(f"  A{idx + 1:03d} ({class_name(idx)}): {val:.4f}")
    print("==========================")


def save_skeleton_video(raw: np.ndarray, output_path: str, title: str) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    skeleton_bone_pairs = (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
        (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
        (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
    )
    edges = [(u - 1, v - 1) for (u, v) in skeleton_bone_pairs]

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title, fontsize=11)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-60)
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Z (Depth)")
    ax.set_zlabel("Y (Height)")

    lines: List = []
    total_frames = int(raw.shape[1])

    def update(frame_idx: int):
        for line in lines:
            line.remove()
        lines.clear()
        for m in range(2):
            pts = raw[:, frame_idx, :, m]  # (3, 25)
            if np.all(pts == 0):
                continue
            xs, ys, zs = pts[0], pts[1], pts[2]
            for u, v in edges:
                line = ax.plot(
                    [xs[u], xs[v]],
                    [zs[u], zs[v]],
                    [ys[u], ys[v]],
                    color="b" if m == 0 else "r",
                    marker="o",
                    markersize=2.5,
                )[0]
                lines.append(line)
        return lines

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    ani.save(output_path, writer="ffmpeg", fps=20)
    plt.close(fig)
    print(f"skeleton_video_output: {output_path}")


def find_gt_preview_video(
    skeleton_path: str, gt: Optional[int], rgb_dir: str
) -> Optional[str]:
    if not os.path.isdir(rgb_dir):
        return None

    # Prefer matching RGB file for the exact same sequence.
    base = os.path.splitext(os.path.basename(skeleton_path))[0]
    exact = os.path.join(rgb_dir, f"{base}_rgb.avi")
    if os.path.exists(exact):
        return exact

    # Fallback to any RGB example from same class.
    if gt is None:
        return None
    class_tag = f"A{gt + 1:03d}"
    pattern = os.path.join(rgb_dir, f"*{class_tag}_rgb.avi")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def overlay_preview_in_corner(
    skeleton_video_path: str, preview_video_path: str, output_path: str
) -> bool:
    # Add a small preview inset in bottom-right.
    overlay_filter = (
        "[1:v]scale=iw*0.28:ih*0.28,drawbox=color=white@0.9:thickness=2[preview];"
        "[0:v][preview]overlay=W-w-16:H-h-16:shortest=1[outv]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        skeleton_video_path,
        "-stream_loop",
        "-1",
        "-i",
        preview_video_path,
        "-filter_complex",
        overlay_filter,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:
        print(f"Warning: failed to overlay GT preview video ({exc}).")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run action recognition for a given skeleton file and print metrics."
    )
    parser.add_argument("--skeleton", required=True, help="Path to input .skeleton file")
    parser.add_argument("--checkpoint", default="checkpoints/ntu60_xs_joint_dste.pth.tar")
    parser.add_argument("--dataset-dir", default="dataset/nturgb+d_skeletons")
    parser.add_argument("--rgb-dir", default="dataset/nturgb+d_rgb")
    parser.add_argument("--gallery-size", type=int, default=300)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--video-output", default="prediction.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    skeleton_path = resolve_cli_path(THIS_DIR, args.skeleton)
    checkpoint_path = resolve_cli_path(THIS_DIR, args.checkpoint)
    dataset_dir = resolve_cli_path(THIS_DIR, args.dataset_dir)
    rgb_dir = resolve_cli_path(THIS_DIR, args.rgb_dir)
    if os.path.isabs(args.video_output):
        video_output = args.video_output
    elif args.video_output == "prediction.mp4":
        video_output = os.path.abspath(os.path.join(THIS_DIR, args.video_output))
    else:
        video_output = os.path.abspath(args.video_output)

    if not os.path.exists(skeleton_path):
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model, has_classifier = load_model(checkpoint_path, device)
    jt, js, raw = preprocess_skeleton_file(skeleton_path)
    gt = parse_label_from_filename(skeleton_path)

    if has_classifier:
        logits = infer_classifier(model, jt, js, device)[0]
        probs = torch.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).item())
        score = float(probs[pred].item())
        topn = min(args.topk, probs.numel())
        vals, idxs = torch.topk(probs, k=topn)
        ranked = [(int(i.item()), float(v.item())) for v, i in zip(vals, idxs)]
        print_metrics(skeleton_path, "classifier", pred, score, gt, ranked)
    else:
        gallery_feats, gallery_labels = build_gallery(
            model=model,
            dataset_dir=dataset_dir,
            query_path=skeleton_path,
            gallery_size=args.gallery_size,
            seed=args.seed,
            device=device,
        )
        query_feat = infer_feature(model, jt, js, device)
        sims = F.cosine_similarity(query_feat, gallery_feats)
        # Weighted-kNN class voting is more stable than single nearest neighbor.
        k_neighbors = max(1, min(args.knn_k, int(sims.numel())))
        n_vals, n_idxs = torch.topk(sims, k=k_neighbors)

        class_scores: Dict[int, float] = {}
        for v, i in zip(n_vals.tolist(), n_idxs.tolist()):
            label = int(gallery_labels[int(i)].item())
            class_scores[label] = class_scores.get(label, 0.0) + float(v)

        ranked_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        pred = ranked_classes[0][0]
        score = ranked_classes[0][1] / k_neighbors
        ranked = ranked_classes[: max(1, args.topk)]
        print_metrics(skeleton_path, "knn_feature", pred, score, gt, ranked)

    gt_text = f"A{gt + 1:03d} ({class_name(gt)})" if gt is not None else "N/A"
    title = f"GT: {gt_text} | Pred: A{pred + 1:03d} ({class_name(pred)})"
    tmp_skeleton_video = video_output + ".skeleton_only.mp4"
    save_skeleton_video(raw, tmp_skeleton_video, title)

    gt_preview = find_gt_preview_video(skeleton_path, gt, rgb_dir)
    if gt_preview is None:
        os.replace(tmp_skeleton_video, video_output)
        print("gt_preview_video: not found; exported skeleton-only video.")
    else:
        ok = overlay_preview_in_corner(tmp_skeleton_video, gt_preview, video_output)
        if ok:
            os.remove(tmp_skeleton_video)
            print(f"gt_preview_video: {gt_preview}")
            print(f"video_output: {video_output}")
        else:
            os.replace(tmp_skeleton_video, video_output)
            print("gt_preview_video: overlay failed; exported skeleton-only video.")
            print(f"video_output: {video_output}")


if __name__ == "__main__":
    main()
