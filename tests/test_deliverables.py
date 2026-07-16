#!/usr/bin/env python3
"""AnyDexRT 三件套交付评估."""
import json, sys, os, subprocess
from pathlib import Path
import numpy as np
import torch

# ── 常量和配置 ────────────────────────────────────────────────
BASELINE_CKPT = "checkpoint/custom_right_last"
NULLSPACE_CKPT = "checkpoint/custom_right_2026-07-15_16-44-15_nullspace"
HTS_PATH = "data/hts_right.npy"
URDF_PATH = "assets/custom_right/URDF_R.urdf"

JOINT_ORDER = [
    'F1-R-MCP2','F1-R-MCP1','F1-R-PIP','F1-R-DIP',
    'F2-R-MCP2','F2-R-MCP1','F2-R-PIP','F2-R-DIP',
    'F3-R-MCP2','F3-R-MCP1','F3-R-PIP','F3-R-DIP',
    'F4-R-MCP2','F4-R-MCP1','F4-R-PIP','F4-R-DIP',
    'F5-R-MCP2','F5-R-MCP1','F5-R-PIP','F5-R-DIP',
]

# From SAPIEN hand.get_joint_limit() — physical radians
JOINT_LOWER = np.array([
    -0.3491, -0.52, -0.35, -0.35, -0.61, -0.52, 0.0, 0.0,
    -0.61, -0.52, 0.0, 0.0, -0.61, -0.52, 0.0, 0.0,
    -0.61, -0.52, 0.0, 0.0], dtype=np.float32)
JOINT_UPPER = np.array([
    0.61, 0.785, 1.31, 1.31, 0.61, 1.57, 1.92, 1.22,
    0.61, 1.57, 1.92, 1.22, 0.61, 1.57, 1.92, 1.22,
    0.61, 1.57, 1.92, 1.22], dtype=np.float32)

FINGER_GROUPS = [
    {'finger':'thumb','keypoint_indices':[0],'joint_indices':[0,1,2,3]},
    {'finger':'index','keypoint_indices':[1],'joint_indices':[4,5,6,7]},
    {'finger':'middle','keypoint_indices':[2],'joint_indices':[8,9,10,11]},
    {'finger':'ring','keypoint_indices':[3],'joint_indices':[12,13,14,15]},
    {'finger':'pinky','keypoint_indices':[4],'joint_indices':[16,17,18,19]},
]
FINGER_NAMES = ['thumb','index','middle','ring','pinky']
FINGER_LABELS = ['Thumb','Index','Middle','Ring','Pinky']
FINGER_START = [0,4,8,12,16]  # joint index for MCP2 of each finger
HUMAN_IDS = [4,8,12,16,20]

# ── 加载模型 ──────────────────────────────────────────────────
from geort.model import IKModel
from geort.analytic_fk import AnalyticFK
from geort.keypoint_normalization import normalize_finger_points

def load_checkpoint(ckpt_dir):
    ckpt = Path(ckpt_dir)
    ik = IKModel(finger_groups=FINGER_GROUPS, n_total_joint=20).cuda()
    ik.load_state_dict(torch.load(ckpt / "last.pth", map_location="cuda"))
    ik.eval()
    with open(ckpt / "normalization.json") as f:
        norm = json.load(f)
    human_stats = norm["human"]
    robot_stats = norm["robot"]
    fk = AnalyticFK(URDF_PATH, JOINT_LOWER, JOINT_UPPER)
    fk.eval()
    return ik, fk, human_stats, robot_stats

print("Loading models...")
ik_b, fk_b, hum_b, rob_b = load_checkpoint(BASELINE_CKPT)
ik_n, fk_n, hum_n, rob_n = load_checkpoint(NULLSPACE_CKPT)

# ── 测试数据 ──────────────────────────────────────────────────
hts = np.load(HTS_PATH)  # [N, 21, 3]
rng = np.random.RandomState(42)
N_eval = 1000
idx = rng.choice(len(hts), N_eval, replace=False)
test_frames = hts[idx]

tips_metric = test_frames[:, HUMAN_IDS, :3].astype(np.float32)
tips_norm_b = normalize_finger_points(tips_metric, FINGER_NAMES, hum_b)
tips_norm_n = normalize_finger_points(tips_metric, FINGER_NAMES, hum_n)

tips_norm_b_t = torch.from_numpy(tips_norm_b).float().cuda()
tips_norm_n_t = torch.from_numpy(tips_norm_n).float().cuda()
tips_metric_t = torch.from_numpy(tips_metric).float().cuda()

@torch.no_grad()
def compute_tip_error(ik, fk, tips_norm, tips_metric):
    joint_norm = ik(tips_norm)
    tips_pred = fk(joint_norm)
    diff = tips_metric - tips_pred
    err_mm = torch.norm(diff, dim=-1) * 1000
    return err_mm.cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# ① 指尖误差
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("① Fingertip Error: Nullspace vs Baseline (synergy=0)")
print("=" * 60)

err_b = compute_tip_error(ik_b, fk_b, tips_norm_b_t, tips_metric_t)
err_n = compute_tip_error(ik_n, fk_n, tips_norm_n_t, tips_metric_t)

print(f"\n{'Finger':>10} {'Baseline mm':>14} {'Nullspace mm':>14} {'Δ%':>8}")
print("-" * 54)
total_b, total_n = 0.0, 0.0
passed = True
for fi in range(5):
    eb = err_b[:, fi].mean()
    en = err_n[:, fi].mean()
    dpct = (en - eb) / eb * 100 if eb > 1e-6 else 0
    total_b += eb; total_n += en
    ok = abs(dpct) < 5
    if not ok: passed = False
    print(f"{FINGER_LABELS[fi]:>10} {eb:>14.3f} {en:>14.3f} {dpct:>+7.2f}%  {'✓' if ok else '✗'}")

print("-" * 54)
tdpct = (total_n - total_b) / total_b * 100
print(f"{'MEAN':>10} {total_b/5:>14.3f} {total_n/5:>14.3f} {tdpct:>+7.2f}%  {'PASS ✓' if passed else 'FAIL ✗'}")

# ═══════════════════════════════════════════════════════════════
# ② 合成轨迹四关节响应表
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② Synthetic Sweep: 4-Joint Response (deg)")
print("=" * 60)

jl_t = torch.from_numpy(JOINT_LOWER).float().cuda()
ju_t = torch.from_numpy(JOINT_UPPER).float().cuda()
half_t = (ju_t - jl_t) / 2.0

def joint_norm_to_phys(joints_norm):
    return jl_t + (joints_norm + 1.0) * half_t

@torch.no_grad()
def sweep_response(ik_model, sweep_finger_idx, n_steps=10):
    x = torch.zeros(n_steps, 5, 3, device='cuda')
    z_vals = torch.linspace(1.0, -1.0, n_steps, device='cuda')
    x[:, sweep_finger_idx, 2] = z_vals
    joints_norm = ik_model(x)
    joints_phys = joint_norm_to_phys(joints_norm)
    jrange_rad = joints_phys.max(dim=0).values - joints_phys.min(dim=0).values
    return np.degrees(jrange_rad.cpu().numpy())

JOINT_SUBNAMES = ["MCP2", "MCP1", "PIP", "DIP"]

for label, ik_m in [("Baseline (syn=0)", ik_b), ("Nullspace 0.01", ik_n)]:
    print(f"\n--- {label} ---")
    header = f"{'Sweep':>8}"
    for fl in FINGER_LABELS:
        for js in JOINT_SUBNAMES:
            header += f" {fl[:3]:>5}-{js}"
    print(header)

    for fi_sweep in range(5):
        jrange_deg = sweep_response(ik_m, fi_sweep)
        row = f"{FINGER_LABELS[fi_sweep]:>8}"
        for fj in range(5):
            base = FINGER_START[fj]
            for jj in range(4):
                row += f" {jrange_deg[base+jj]:>8.1f}"
        print(row)

    print(f"\n  Non-swept MCP2 response (°):")
    mcp2_all = []
    for fi_sweep in range(5):
        jrange_deg = sweep_response(ik_m, fi_sweep)
        others = [jrange_deg[FINGER_START[fj]] for fj in range(5) if fj != fi_sweep]
        mcp2_all.extend(others)
        print(f"    Sweep {FINGER_LABELS[fi_sweep]:>6}: non-swept MCP2 = [{', '.join(f'{v:.1f}°' for v in others)}]")
    print(f"    GLOBAL mean non-swept MCP2 = {np.mean(mcp2_all):.2f}°")

# ═══════════════════════════════════════════════════════════════
# ③ 启动跨 seed 训练
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ Cross-Seed Training — Launching 4 runs...")
print("=" * 60)

from geort.trainer import GeoRTTrainer, resolve_human_training_input
from geort.utils.config_utils import get_config

config = get_config('custom_right')
hp = resolve_human_training_input(HTS_PATH)

for seed in [42, 123]:
    for variant, nw, sw in [("syn0", 0.0, 0.0), ("null", 0.01, 0.0)]:
        logf = f"/tmp/train_seed{seed}_{variant}.log"
        cmd = (
            f"cd {os.getcwd()} && rm -rf checkpoint/custom_right_*_seed{seed}_{variant} 2>/dev/null && "
            f"PYTHONPATH=. {sys.executable} -c \""
            f"import torch; torch.manual_seed({seed}); "
            f"import numpy as np; np.random.seed({seed}); "
            f"import random; random.seed({seed}); "
            f"from geort.trainer import GeoRTTrainer, resolve_human_training_input; "
            f"from geort.utils.config_utils import get_config; "
            f"config=get_config('custom_right'); "
            f"t=GeoRTTrainer(config); "
            f"hp=resolve_human_training_input('{HTS_PATH}'); "
            f"t.train(hp, tag='seed{seed}_{variant}', fk_backend='analytic', epoch=200, "
            f"w_chamfer=1.0, w_distance=1.0, w_curvature=0.1, w_motion=1.0, "
            f"nullspace_weight={nw}, synergy_weight={sw}, "
            f"w_pinch=0.0, w_collision=0.0, w_mcp1_fist_prior=0.0, "
            f"save_every=0, chamfer_target='uniform', update_latest=False); "
            f"print('DONE')\" > {logf} 2>&1 &"
        )
        subprocess.Popen(cmd, shell=True, executable='/bin/bash',
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ✓ seed={seed} {variant} → {logf}")

print("\n✓ All 4 launched. After completion, run part-③ std report.")
