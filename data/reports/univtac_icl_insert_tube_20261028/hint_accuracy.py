import sys; sys.path.insert(0,"."); from prometheus.policy.univtac_icl import jam_guard
import json,glob,sys
from pathlib import Path
S=json.load(open('data/reports/univtac_icl_insert_tube_20261028/summary.json'))
R=Path('data/runs/univtac_icl')
for g,base in [('G1','insert_tube_guardG1_visuotactile_20261028'),('G2','insert_tube_guardG2_visuotactile_20261028'),('G3','insert_tube_guardG3_visual_20261028')]:
    name=[k for k in S if k.startswith(g)][0]
    for seed,r in S[name].items():
        runs=[base+'_infra1'] if (g=='G2' and seed=='1000006') else [base]
        cwd=R/runs[0]/f'seed_{seed}'
        ev=[json.loads(l) for l in open(cwd/'audit/events.jsonl')]
        nx,ny=r['noise_mm']
        firstjam=None; hx=hy=0; sh=[]
        for e in ev:
            if e['event']!='executed': continue
            s1=e['receipt'].get('system1',{})
            if e['receipt'].get('stop_reason')=='s1_jam':
                ms=s1.get('marker_shear_px')
                if firstjam is None: firstjam=ms
                h=s1.get('offset_hint')
                if h:
                    hx+= (h['shift_hand_x']=='-')==(nx>0); hy+=(h['shift_hand_y']=='+')==(ny<0); sh.append(h['shift_hand_x']+h['shift_hand_y'])
        print(g,seed,r['status'][0],'noise',r['noise_mm'],'calls',r['calls'],'jams',r['jams'],'end',r['end'],'hint_ok x %d y %d /%d'%(hx,hy,len(sh)),''.join(sh[:6]),'first',firstjam)
print('---')
import collections
tot=collections.Counter()
for g,base in [('G1','insert_tube_guardG1_visuotactile_20261028'),('G2','insert_tube_guardG2_visuotactile_20261028'),('G3','insert_tube_guardG3_visual_20261028')]:
    name=[k for k in S if k.startswith(g)][0]
    for seed,r in S[name].items():
        cwd=R/((base+'_infra1') if (g=='G2' and seed=='1000006') else base)/f'seed_{seed}'
        nx,ny=r['noise_mm']; k=0
        for e in map(json.loads,open(cwd/'audit/events.jsonl')):
            if e['event']=='executed' and e['receipt'].get('stop_reason')=='s1_jam':
                h=e['receipt'].get('system1',{}).get('offset_hint') or (jam_guard.offset_hint(e['receipt']['system1']['marker_shear_px']) if 'marker_shear_px' in e['receipt'].get('system1',{}) else None)
                if h is None: continue
                tag='first' if k==0 else 'later'; k+=1
                tot[(g,tag,'n')]+=1; tot[(g,tag,'x')]+=(h['shift_hand_x']=='-')==(nx>0); tot[(g,tag,'y')]+=(h['shift_hand_y']=='+')==(ny<0)
for g in ['G1','G2','G3']:
    for t in ['first','later']:
        print(g,t,'x %d y %d / %d'%(tot[(g,t,'x')],tot[(g,t,'y')],tot[(g,t,'n')]))
for n in ['v5 direct / zero_shot','v5 direct / visual','v5 direct / visuotactile']:
    print(n, S[n]['1000001']['end'], S[n]['1000001']['calls'])
