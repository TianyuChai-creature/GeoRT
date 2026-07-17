#!/usr/bin/env python3
"""Canonical C2eL_s42 evaluation: D1, unified local-T, anchors and cloud shape."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from geort.model import IKModel
from geort.analytic_fk import AnalyticFK
from geort.keypoint_normalization import normalize_finger_points
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types
from geort.env.hand import HandKinematicModel
from geort.motion_frames import build_human_motion_frames, robot_task_frames

ROOT=Path(__file__).resolve().parents[1]
FINGERS=['thumb','index','middle','ring','pinky']; IDS=[4,8,12,16,20]
GROUPS=[{'finger':f,'keypoint_indices':[i],'joint_indices':list(range(4*i,4*i+4))} for i,f in enumerate(FINGERS)]

def sha(path):
 d=hashlib.sha256()
 with open(path,'rb') as h:
  for b in iter(lambda:h.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def mean(x): return float(np.mean(x))
def cos_metric(a,b,rh=None,rr=None):
 da=a[1:]-a[:-1]; db=b[1:]-b[:-1]
 if rh is not None:
  da=np.einsum('...ji,...j->...i',rh[:-1],da); db=np.einsum('...ji,...j->...i',rr[:-1],db)
 den=np.linalg.norm(da,axis=-1)*np.linalg.norm(db,axis=-1); valid=den>1e-6
 vals=np.full(den.shape,np.nan); vals[valid]=(da[valid]*db[valid]).sum(-1)/den[valid]
 return [float(np.nanmean(vals[:,i])) for i in range(5)],float(np.nanmean(vals))
def make_model(device, checkpoint):
 m=IKModel(finger_groups=GROUPS,n_total_joint=20).to(device)
 m.load_state_dict(torch.load(checkpoint/'last.pth',map_location=device,weights_only=True)); m.eval(); return m
def predict(model,fk,human,stats,device):
 hn=normalize_finger_points(human,FINGERS,stats['human'])
 outs=[]; tips=[]; rnorm=[]
 with torch.no_grad():
  for start in range(0,len(hn),2048):
   qn=model(torch.from_numpy(hn[start:start+2048]).to(device).float()); tp=fk(qn)
   outs.append(qn.cpu().numpy()); tips.append(tp.cpu().numpy())
 qn=np.concatenate(outs); tips=np.concatenate(tips)
 return hn,qn,tips,normalize_finger_points(tips,FINGERS,stats['robot'])
def local_frames(config,info,robot_norm):
 z=np.load(ROOT/'data/custom_right_with_rot.npz',allow_pickle=True); key=z['keypoint'].item(); lr=z['link_rotation'].item()
 target=np.stack([key[n] for n in info['link']],axis=1).astype(np.float32)
 targetnorm=normalize_finger_points(target,FINGERS,robot_norm)
 task=robot_task_frames(torch.from_numpy(np.stack([lr[n] for n in info['link']],axis=1))).numpy()
 trees=[cKDTree(targetnorm[:,fi]) for fi in range(5)]
 return trees,task
def borrowed_rotations(robot_norm,trees,task):
 out=np.empty((len(robot_norm),5,3,3),np.float32)
 for fi,t in enumerate(trees): out[:,fi]=task[t.query(robot_norm[:,fi])[1],fi]
 return out
def cloud_shape(hn,rn):
 rng=np.random.RandomState(42); n=min(len(hn),50000); idx=rng.choice(len(hn),n,replace=False); h=hn[idx]; r=rn[idx]
 rows=[]
 for fi,name in enumerate(FINGERS):
  a=rng.randint(0,n,10000); b=rng.randint(0,n,10000); dh=np.linalg.norm(h[a,fi]-h[b,fi],axis=1); dr=np.linalg.norm(r[a,fi]-r[b,fi],axis=1); ratio=dr[dh>1e-8]/dh[dh>1e-8]
  sh=np.linalg.svd(h[:,fi]-h[:,fi].mean(0),compute_uv=False); sr=np.linalg.svd(r[:,fi]-r[:,fi].mean(0),compute_uv=False)
  rows.append({'finger':name,'contraction_ratio_p50':float(np.percentile(ratio,50)),'pca_sigma3_ratio_mapped_over_human':float(sr[2]/sh[2])})
 return rows
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--anchor-path',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-md',type=Path,required=True);a=p.parse_args()
 device=torch.device('cuda'); ck=a.checkpoint.resolve(); meta=json.loads((ck/'training_metadata.json').read_text()); norm=json.loads((ck/'normalization.json').read_text())
 config=get_config('custom_right'); info=select_keypoint_types(parse_config_keypoint_info(config),allowed_types=('tip',)); hand=HandKinematicModel.build_from_config(config); lo,hi=(np.asarray(x,np.float32) for x in hand.get_joint_limit()); fk=AnalyticFK(config['urdf_path'],lo,hi,tip_offsets=info['offset']).to(device).eval(); model=make_model(device,ck)
 raw=np.load(ROOT/'data/hts_right.npy').astype(np.float32); rng=np.random.RandomState(42); idx=rng.choice(len(raw),1000,replace=False); human=raw[idx][:,IDS,:]
 hn,qn,tips,rn=predict(model,fk,human,norm,device); err=np.linalg.norm(tips-human,axis=-1); gpf,gall=cos_metric(hn,rn); hf,_=build_human_motion_frames(raw[idx]); trees,task=local_frames(config,info,norm['robot']); rr=borrowed_rotations(rn,trees,task); lpf,lall=cos_metric(hn,rn,hf,rr)
 # fixed open-degree bins in physical hand-base space, all D1 frames; map in batches.
 allhuman=raw[:,IDS,:]; openness=np.linalg.norm(allhuman,axis=-1).mean(1); edges=np.linspace(.074412,.170439,11); ten=[]
 for bi in range(10):
  mask=(openness>=edges[bi]) & ((openness<edges[bi+1]) if bi<9 else (openness<=edges[bi+1])); h=allhuman[mask]; _,_,t,_=predict(model,fk,h,norm,device); e=np.linalg.norm(t-h,axis=-1)
  ten.append({'bin':bi+1,'lower':float(edges[bi]),'upper':float(edges[bi+1]),'count':int(mask.sum()),'tip_error_m_per_finger':[mean(e[:,i]) for i in range(5)]})
 # Cloud metrics use all available D1 rows, capped reproducibly to 50k.
 sample=rng.choice(len(raw),min(50000,len(raw)),replace=False); hcloud=raw[sample][:,IDS,:]; hcn,_,_,rcn=predict(model,fk,hcloud,norm,device); shape=cloud_shape(hcn,rcn)
 an=np.load(a.anchor_path,allow_pickle=True); ctx=an['human_tip_contexts'].astype(np.float32); target=an['robot_points'].astype(np.float32); fi=an['finger_indices']; _,_,atips,_=predict(model,fk,ctx,norm,device); ares=np.linalg.norm(atips[np.arange(len(fi)),fi]-target,axis=-1); anchor_per_finger=[{'finger':name,'count':int((fi==finger).sum()),'mean_m':mean(ares[fi==finger]),'max_m':float(ares[fi==finger].max())} for finger,name in enumerate(FINGERS)]
 result={'protocol':{'d1_frames':1000,'random_state':42,'units':{'tip_error':'m','gmc_lmc':'0_to_1'},'open_bucket_edges':[float(v) for v in edges],'cloud_rows':int(len(hcloud))},'checkpoint':{'path':str(ck.relative_to(ROOT)),'last_pth_sha256':sha(ck/'last.pth'),'git_hash':meta['cli_args']['run_git_commit']},'anchor_bundle':{'path':str(a.anchor_path),'sha256':sha(a.anchor_path)},'d1_tip_error_m':{'per_finger':[mean(err[:,i]) for i in range(5)],'mean':mean(err)},'gmc':{'per_finger':gpf,'overall':gall},'unified_local_t_lmc':{'per_finger':lpf,'overall':lall},'open_degree_bins':ten,'cloud_shape':shape,'anchor_tip_residual_m':{'mean':mean(ares),'max':float(ares.max()),'count':int(len(ares)),'per_finger':anchor_per_finger}}
 a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 lines=['# Canonical Evaluation','', '## Definitions','', '- D1: `RandomState(42)` without replacement, 1000 frames.', '- GMC: global normalized-tip motion cosine.', '- unified local-T LMC: `geort.motion_frames.build_human_motion_frames` plus nearest target rotation in normalized robot-tip space.', '- Open-degree: mean five-TIP distance to hand-base origin, fixed edges `[0.074412, 0.170439]`.', '- Cloud contraction: 10,000 random normalized-space point pairs per finger; PCA: mapped/human third singular value.', '', '## Results','', '```json',json.dumps(result,indent=2,sort_keys=True),'```']
 a.output_md.write_text('\n'.join(lines)+'\n')
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
