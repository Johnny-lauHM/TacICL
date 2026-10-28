"""Analyze probe_multi runs: how well public marker-shear features at a jam predict the hand residual
(privileged label, hole frame) for first vs later jams, per baseline and with/without a retract before
the push. Usage: analyze_multi.py RUN_DIR"""
import glob
import json
import sys

import numpy as np

rows, results = [], []
for path in sorted(glob.glob(sys.argv[1] + '/seed_*/audit/jams.jsonl')):
    seed = path.split('/')[-3]
    res = json.load(open(path.replace('jams.jsonl', 'result.json'))) if glob.glob(path.replace('jams.jsonl', 'result.json')) else {}
    jams = [json.loads(l) for l in open(path)]
    results.append((seed, res.get('check_success'), len(jams), res.get('final', {}).get('hand_res_mm'),
                    [round(v * 1000, 1) for v in res.get('metadata', {}).get('random_noise', [0, 0])[:2]]))
    for j in jams:
        j['seed'] = seed
        rows.append(j)

print('seed success jams final_hand_res noise')
for r in results:
    print(*r)

FEATS = ['lateral_px', 'lateral_left_px', 'lateral_right_px', 'asymmetry_px', 'axial_left_px', 'axial_right_px']


def table(sel, label):
    if len(sel) < 3:
        print(f'{label}: n={len(sel)}')
        return
    hx = np.array([r['priv']['hand_res_mm'][0] for r in sel])
    hy = np.array([r['priv']['hand_res_mm'][1] for r in sel])
    print(f'{label}: n={len(sel)}  |hand x| median {np.median(abs(hx)):.1f}  |hand y| median {np.median(abs(hy)):.1f} mm')
    for base in ('episode', 'push'):
        out = []
        for f in FEATS:
            v = np.array([r[base][f] for r in sel])
            cx, cy = np.corrcoef(v, hx)[0, 1], np.corrcoef(v, hy)[0, 1]
            out.append(f'{f} rx{cx:+.2f} ry{cy:+.2f}')
        print(f'  {base:7s} ' + ' | '.join(out))
    # the guard v1 sign rule (episode baseline)
    lat = np.array([r['episode']['lateral_px'] for r in sel])
    asy = np.array([r['episode']['asymmetry_px'] for r in sel])
    print(f'  v1 rule sign acc: x {np.mean(np.sign(lat) == np.sign(hx)):.2f}  y {np.mean(np.sign(asy) == -np.sign(hy)):.2f}')


table([r for r in rows if r['jam'] == 0], 'first jam')
table([r for r in rows if r['jam'] > 0], 'later jams')
table([r for r in rows if r['jam'] > 0 and r['retracted_before']], 'later, retracted before push')
table([r for r in rows if r['jam'] > 0 and not r['retracted_before']], 'later, no retract')
