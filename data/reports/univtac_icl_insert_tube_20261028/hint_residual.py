"""Hint accuracy against the residual offset at each jam (analysis only, after the fact).

residual (hole frame, mm) = hidden pre_move offset + hand displacement since the first policy observation
projected on the hole axes. The hole frame is the base frame rotated -30 deg about y (insert_tube.py:
add_rotation([0, -pi/6, 0])), so hole x = (cos30, 0, sin30) in base and hole y = base y. In-hand slip is
ignored. Also reports the lateral step the model took after each hint. Usage: hint_residual.py
"""
import json
import math
from collections import Counter
from pathlib import Path

S = json.load(open(Path(__file__).with_name('summary.json')))
R = Path('/home/nus/workspace/PrometheusV4/data/runs/univtac_icl')
HX = (math.cos(math.pi / 6), 0.0, math.sin(math.pi / 6))
GROUPS = [('G1', 'insert_tube_guardG1_visuotactile_20261028'), ('G2', 'insert_tube_guardG2_visuotactile_20261028'),
          ('G3', 'insert_tube_guardG3_visual_20261028')]


def hint_from_shear(f):
    return ('-' if f['lateral_px'] > 0 else '+'), ('+' if f['asymmetry_px'] > 0 else '-')


for g, base in GROUPS:
    name = next(k for k in S if k.startswith(g))
    c, steps = Counter(), []
    for seed, row in S[name].items():
        run = base + '_infra1' if (g == 'G2' and seed == '1000006') else base
        ex = [json.loads(l) for l in open(R / run / f'seed_{seed}' / 'audit' / 'events.jsonl')]
        ex = [e for e in ex if e['event'] == 'executed']
        if not ex:
            continue
        nx, ny = row['noise_mm']
        ee0 = ex[0]['before']['ee_pos_mm']
        for i, e in enumerate(ex):
            r = e['receipt']
            if r.get('stop_reason') != 's1_jam':
                continue
            s1 = r.get('system1', {})
            h = s1.get('offset_hint')
            h = (h['shift_hand_x'], h['shift_hand_y']) if h else (hint_from_shear(s1['marker_shear_px']) if 'marker_shear_px' in s1 else None)
            if h is None:
                continue
            d = [a - b for a, b in zip(r['result_state']['ee_pos_mm'], ee0)]
            rx = nx + sum(a * b for a, b in zip(d, HX))
            ry = ny + d[1]
            tag = 'first' if c[(seed, 'k')] == 0 else 'later'
            c[(seed, 'k')] += 1
            c[tag, 'n'] += 1
            c[tag, 'x'] += (h[0] == '-') == (rx > 0)
            c[tag, 'y'] += (h[1] == '+') == (ry < 0)
            if abs(ry) < 1.0:
                c[tag, 'y_small'] += 1
            if i + 1 < len(ex):
                nxt = ex[i + 1]
                dd = [a - b for a, b in zip(nxt['receipt']['result_state']['ee_pos_mm'], r['result_state']['ee_pos_mm'])]
                steps.append((abs(sum(a * b for a, b in zip(dd, HX))), abs(dd[1]), abs(rx), abs(ry)))
    print(g, ' '.join(f"{t}: x {c[t,'x']}/{c[t,'n']} y {c[t,'y']}/{c[t,'n']} (|res y|<1mm {c[t,'y_small']})" for t in ('first', 'later')))
    if steps:
        sx = sorted(s[0] for s in steps); sy = sorted(s[1] for s in steps)
        rxs = sorted(s[2] for s in steps); rys = sorted(s[3] for s in steps)
        print(f"   step after hint median |dx_hole| {sx[len(sx)//2]:.1f} |dy| {sy[len(sy)//2]:.1f} mm; residual median |x| {rxs[len(rxs)//2]:.1f} |y| {rys[len(rys)//2]:.1f} mm")
