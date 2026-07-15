# FK Backend Ablation

Evaluation: 1000 random frames (seed=42) from `data/hts_right.npy` (234114 total).
Coverage verified against full dataset: 88–100% per axis.
GeoRT baseline (main branch `df031c3`) could not be evaluated — see [Baseline Incompatibility](#baseline-incompatibility) below.

## Motion Consistency (LMC/GMC)

GMC = cosine similarity of human vs robot consecutive-frame fingertip displacement
directions in the shared normalised coordinate frame.  LMC = GMC (T=I identity
local frame; Step 4 local coordinate frames pending).

| Finger | Analytic FK GMC | Neural FK GMC |
|--------|----------------|---------------|
| thumb  | 0.9924 | 0.9915 |
| index  | 0.9941 | 0.9947 |
| middle | 0.9901 | 0.9949 |
| ring   | 0.9978 | **0.5141** |
| pinky  | 0.9957 | **0.5185** |
| **OVERALL** | **0.9940** | **0.8030** |

Valid consecutive frame pairs: 99.3% (tiny-displacement frames filtered at 1e-6 m).

## Joint-Space Metrics

| Backend | Signed Gain | Rest Offset | Sat (any) |
|---------|------------|-------------|-----------|
| GeoRT baseline (main) | — | — | — |
| Neural FK | 0.925 | 16.9% | 17.5% |
| Analytic FK (ours) | 0.872 | 17.8% | 15.1% |

### Metric Definitions

**Signed Gain** = median over joints of `(qpos_max - qpos_min) / (joint_upper - joint_lower)`.
Evaluated on 1000 random frames.  ~1.0 ideal (joint range fully utilised).

**Rest Offset** = median over joints of `|qpos_j - q_default_j| / (joint_upper_j - joint_lower_j)`,
where q_default = (lower + upper) / 2.  Lower is better; above 5% indicates
systematic bias away from the centre of the joint range.

**Saturation** = fraction of frames where ANY joint is within 5% of its
lower or upper limit (margin = 0.05 × joint_range).  0% ideal.
Joints with ≥5% saturation in either checkpoint:

| Joint | Analytic FK | Neural FK |
|-------|------------|-----------|
| F1-R-MCP2 (thumb abduction) | lo=3.9% hi=3.1% | lo=5.0% |
| F2-R-PIP (index flexion) | lo=9.0% hi=5.8% | — |
| F2-R-DIP (index flexion) | hi=10.9% | — |
| F3-R-PIP (middle flexion) | lo=11.5% | — |
| F3-R-DIP (middle flexion) | hi=9.8% | — |
| F4-R-PIP (ring flexion) | hi=7.3% | — |
| F5-R-MCP1 (pinky flexion) | hi=8.5% | — |
| F4-R-DIP (ring flexion) | — | hi=10.0% |
| F5-R-MCP2 (pinky abduction) | — | lo=7.7% |

The analytic FK checkpoint saturates more flexion joints (PIP/DIP at upper limits
→ fingers extended straight), while the neural FK shows fewer saturation patterns
but has the ring DIP and pinky abduction at limits.

## Training Configuration

Both AnyDexRT checkpoints: chamfer=1.0, distance=1.0, curvature=0.1, motion=1.0,
motion_delta=0.01, pinch=0, collision=0, mcp1_fist_prior=0.  200 epochs.
Human-shaped target cloud disabled (uniform sampling).  `fk_backend` varies.

## Baseline Incompatibility

Attempting to train/evaluate the GeoRT main-branch checkpoint (`df031c3`)
failed because:

1. **Data path mismatch**: main branch resolves `-human_data human` →
   `data/human.npy`, which requires a symlink from `hts_right.npy`.
2. **10-keypoint pipeline**: main expects 10 keypoints (5 tip + 5 PIP) from
   config; our data provides only tip landmarks (human_id [4,8,12,16,20]),
   and the PIP keypoint extraction (`human_ids` = 10 indices) fails.
3. **FK model architecture mismatch**: main FK outputs 10 keypoints;
   retraining requires the full 10-keypoint target cloud which was cleared
   from `data/`.

**Error signature**: training hangs during FK model pretraining (SAPIEN
initialisation loop for 100K kinematics samples), never reaches IK loop.
No explicit Python exception — the process blocks on SAPIEN's
`compute_forward_kinematics` after several hundred iterations.

**Recommendation**: either restore the cleared `data/custom_right.npz`
target cloud and PIP landmark data, or keep the baseline row as N/A
and focus on the neural-vs-analytic FK comparison which already provides
a controlled ablation.
