from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path('/home/creature/Desktop/GeoRT/.worktrees/AnyDexRT')
OUT = ROOT / 'outputs' / 'c2_variants' / 'motion_consistency_eval'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('ev', ROOT / 'outputs/final_matrix/evaluate_final_matrix.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

TAGS = ('c2b_s42', 'c2b_s123', 'c2e_s42', 'c2e_s123')
ARCHIVE_TAGS = ('c0_s42', 'c0_s123', 'c2_s42', 'c2_s123')
ALL_TAGS = ARCHIVE_TAGS + TAGS
F = ev.FINGERS
IDS = ev.IDS

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def setup():
    config = ev.get_config('custom_right')
    info = ev.select_keypoint_types(ev.parse_config_keypoint_info(config), allowed_types=('tip',))
    hand = ev.HandKinematicModel.build_from_config(config)
    lo, hi = (np.asarray(x, np.float32) for x in hand.get_joint_limit())
    fk = ev.AnalyticFK(config['urdf_path'], lo, hi, tip_offsets=info['offset']).to(ev.DEV).eval()
    raw = np.load(ROOT / 'data/hts_right.npy').astype(np.float32)
    idx = np.random.RandomState(42).choice(len(raw), 1000, replace=False)
    frames = raw[idx]
    human = frames[:, IDS, :]
    z = np.load(ROOT / 'data/custom_right_with_rot.npz', allow_pickle=True)
    keypoint = z['keypoint'].item()
    link_rotation = z['link_rotation'].item()
    metric = np.stack([keypoint[name] for name in info['link']], axis=1).astype(np.float32)
    task_frames = ev.robot_task_frames(torch.from_numpy(np.stack([link_rotation[name] for name in info['link']], axis=1))).numpy()
    human_frames, _ = ev.build_human_motion_frames(frames)
    return info, lo, hi, fk, frames, human, metric, task_frames, human_frames

def predict_metrics(tag, info, fk, human, metric, task_frames, human_frames):
    directory, model, norm, metadata = ev.load(tag)
    _, tips, _, robot_norm = ev.predict(model, fk, human, norm)
    human_norm = ev.normalize_finger_points(human, F, norm['human'])
    _, gmc = ev.cos_metric(human_norm, robot_norm)
    robot_target_norm = ev.normalize_finger_points(metric, F, norm['robot'])
    borrowed = np.empty((len(human), 5, 3, 3), dtype=np.float32)
    for finger in range(5):
        _, nearest = cKDTree(robot_target_norm[:, finger]).query(robot_norm[:, finger])
        borrowed[:, finger] = task_frames[nearest, finger]
    lpf, lmc = ev.cos_metric(human_norm, robot_norm, human_frames, borrowed)
    return {
        'checkpoint': str(directory.relative_to(ROOT)),
        'checkpoint_last_pth_sha256': sha256(directory / 'last.pth'),
        'gmc_overall': float(gmc),
        'lmc_local_per_finger': [float(x) for x in lpf],
        'lmc_local_overall': float(lmc),
        'metadata_git_commit': metadata.get('git_commit', metadata.get('run_git_commit')),
    }, model, norm

def perturb_c2b(info, fk, frames, human, metric, task_frames, human_frames, baseline):
    _, model, norm, _ = ev.load('c2b_s42')
    out = []
    for axis in 'XYZ':
        for degree in (-15.0, -5.0, 5.0, 15.0):
            perturbed_frames = frames @ ev.rot(axis, degree).T
            perturbed_human = perturbed_frames[:, IDS, :]
            _, tips, _, robot_norm = ev.predict(model, fk, perturbed_human, norm)
            errors = np.linalg.norm(tips - perturbed_human, axis=-1)
            human_norm = ev.normalize_finger_points(perturbed_human, F, norm['human'])
            _, gmc = ev.cos_metric(human_norm, robot_norm)
            target_norm = ev.normalize_finger_points(metric, F, norm['robot'])
            borrowed = np.empty((len(perturbed_human), 5, 3, 3), dtype=np.float32)
            for finger in range(5):
                _, nearest = cKDTree(target_norm[:, finger]).query(robot_norm[:, finger])
                borrowed[:, finger] = task_frames[nearest, finger]
            _, lmc = ev.cos_metric(human_norm, robot_norm, human_frames, borrowed)
            out.append({
                'axis': axis,
                'degree': degree,
                'tip_error_m_per_finger': [float(np.mean(errors[:, i]) - baseline['tip_error_m_per_finger'][i]) for i in range(5)],
                'tip_error_m_mean_delta': float(np.mean(errors) - baseline['tip_error_m_mean']),
                'lmc_local_delta': float(lmc - baseline['lmc_local_overall']),
                'gmc_delta': float(gmc - baseline['gmc_overall']),
            })
    return out

def fmt(v):
    return f'{v:.9g}'

def main():
    info, lo, hi, fk, frames, human, metric, task_frames, human_frames = setup()
    archived_final = json.loads((ROOT / 'outputs/final_matrix/final_matrix.json').read_text())
    archived_supplement = json.loads((ROOT / 'outputs/final_matrix/supplement/supplement.json').read_text())

    fresh = {}
    for tag in ALL_TAGS:
        fresh[tag], _, _ = predict_metrics(tag, info, fk, human, metric, task_frames, human_frames)

    gmc = {}
    lmc = {}
    self_checks = {}
    for tag in ARCHIVE_TAGS:
        archived_run = archived_final['part_d']['runs'][tag]
        archived_local = archived_supplement['lmc_unified_local'][tag]
        gmc[tag] = {'value': float(archived_run['gmc_overall']), 'source': 'archive'}
        lmc[tag] = {
            'per_finger': [float(x) for x in archived_local['lmc_local_per_finger']],
            'overall': float(archived_local['lmc_local_overall']),
            'source': 'archive',
        }
        self_checks[tag] = {
            'gmc_max_abs_difference': abs(fresh[tag]['gmc_overall'] - gmc[tag]['value']),
            'lmc_max_abs_difference': float(max(np.max(np.abs(np.asarray(fresh[tag]['lmc_local_per_finger']) - np.asarray(lmc[tag]['per_finger']))), abs(fresh[tag]['lmc_local_overall'] - lmc[tag]['overall']))),
        }
    for tag in TAGS:
        gmc[tag] = {'value': fresh[tag]['gmc_overall'], 'source': 'recomputed'}
        lmc[tag] = {'per_finger': fresh[tag]['lmc_local_per_finger'], 'overall': fresh[tag]['lmc_local_overall'], 'source': 'recomputed'}

    baseline = {
        'tip_error_m_per_finger': archived_final['part_d']['runs']['c2_s42']['tip_error_m_per_finger'],
        'tip_error_m_mean': archived_final['part_d']['runs']['c2_s42']['tip_error_m_mean'],
        'gmc_overall': float(archived_final['part_d']['runs']['c2_s42']['gmc_overall']),
        'lmc_local_overall': float(archived_supplement['lmc_unified_local']['c2_s42']['lmc_local_overall']),
    }
    c2b_perturb = perturb_c2b(info, fk, frames, human, metric, task_frames, human_frames, baseline)
    archive_perturb = {tag: archived_final['part_e'][tag] for tag in ('c0_s42', 'c2_s42')}
    max_item = max(
        (dict(row, finger=F[i], tip_delta_m=row['tip_error_m_per_finger'][i], abs_tip_delta_m=abs(row['tip_error_m_per_finger'][i])) for row in c2b_perturb for i in range(5)),
        key=lambda r: r['abs_tip_delta_m'],
    )

    command = '/home/creature/Desktop/GeoRT/.venv/bin/python /tmp/evaluate_c2be_motion.py --device cuda --frames 1000 --random-state 42 --perturbations X/Y/Z:{-15,-5,+5,+15}'
    data = {
        'base_commit': '4827adb',
        'command': command,
        'protocol': {'d1_frames': 1000, 'random_state': 42, 'units_tip_error': 'm'},
        'checkpoint_sha256': {tag: fresh[tag]['checkpoint_last_pth_sha256'] for tag in ALL_TAGS},
        'gmc': gmc,
        'lmc_unified_local': lmc,
        'archive_recompute_max_abs_difference': self_checks,
        'part_e_c2b_s42': c2b_perturb,
        'part_e_archive_c0_s42_c2_s42': archive_perturb,
        'c2b_s42_worst_abs_tip_delta': max_item,
    }
    (OUT / 'c2be_motion_consistency.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

    lines = [
        '# C2b/C2e motion consistency and perturbation evaluation', '',
        f'base commit: `{data["base_commit"]}`', '',
        '## GMC', '',
        f'command: `{command}`', '',
        '| checkpoint | source | GMC overall | SHA256(last.pth) |',
        '|---|---:|---:|---|',
    ]
    for tag in ALL_TAGS:
        lines.append(f'| {tag} | {gmc[tag]["source"]} | {fmt(gmc[tag]["value"])} | {fresh[tag]["checkpoint_last_pth_sha256"]} |')
    lines += ['', '## Unified local-T LMC', '', f'command: `{command}`', '', '| checkpoint | source | thumb | index | middle | ring | pinky | overall | SHA256(last.pth) |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for tag in ALL_TAGS:
        row = lmc[tag]
        lines.append('| ' + tag + ' | ' + row['source'] + ' | ' + ' | '.join(fmt(v) for v in row['per_finger']) + f' | {fmt(row["overall"])} | {fresh[tag]["checkpoint_last_pth_sha256"]} |')
    lines += ['', '| archive-recompute self-check | GMC max abs | LMC max abs |', '|---|---:|---:|']
    for tag in ARCHIVE_TAGS:
        lines.append(f'| {tag} | {fmt(self_checks[tag]["gmc_max_abs_difference"])} | {fmt(self_checks[tag]["lmc_max_abs_difference"])} |')
    lines += ['', '## Part E rigid input perturbation, D1 RandomState(42), 1000 frames', '', f'command: `{command}`', '', '| axis | deg | c0 thumb Δm | c0 index Δm | c0 middle Δm | c0 ring Δm | c0 pinky Δm | c0 mean Δm | c2 thumb Δm | c2 index Δm | c2 middle Δm | c2 ring Δm | c2 pinky Δm | c2 mean Δm | c2b thumb Δm | c2b index Δm | c2b middle Δm | c2b ring Δm | c2b pinky Δm | c2b mean Δm |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    by_archive = {tag: {(x['axis'], float(x['degree'])): x for x in rows} for tag, rows in archive_perturb.items()}
    for row in c2b_perturb:
        key = (row['axis'], float(row['degree']))
        c0 = by_archive['c0_s42'][key]
        c2 = by_archive['c2_s42'][key]
        values = c0['tip_error_m_per_finger'] + [c0['tip_error_m_mean_delta']] + c2['tip_error_m_per_finger'] + [c2['tip_error_m_mean_delta']] + row['tip_error_m_per_finger'] + [row['tip_error_m_mean_delta']]
        lines.append(f'| {row["axis"]} | {row["degree"]:+.0f} | ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines += ['', f'c2b_s42 SHA256(last.pth): `{fresh["c2b_s42"]["checkpoint_last_pth_sha256"]}`', '', '| c2b worst tip Δ | axis | deg | finger | signed Δm | abs Δm |', '|---|---:|---:|---|---:|---:|', f'| c2b_s42 | {max_item["axis"]} | {max_item["degree"]:+.0f} | {max_item["finger"]} | {fmt(max_item["tip_delta_m"])} | {fmt(max_item["abs_tip_delta_m"])} |']
    (OUT / 'c2be_motion_consistency.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
