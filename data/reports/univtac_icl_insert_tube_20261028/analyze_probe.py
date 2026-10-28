"""Summarize jam_probe runs: per seed, hidden offset, where the straight push stalls, tactile features
at the stall and the privileged re-alignment the expert needed. Usage: analyze_probe.py RUN_DIR"""
import glob, json, sys
import numpy as np

for path in sorted(glob.glob(sys.argv[1] + '/seed_*/audit/steps.jsonl')):
    seed = path.split('/')[-3]
    rows = [json.loads(l) for l in open(path)]
    res = json.load(open(path.replace('steps.jsonl', 'result.json')))
    noise = np.round(np.array(res['metadata']['random_noise'][:2]) * 1000, 1)
    jam = [r for r in rows if r['phase'] == 'jam']
    re_ = [r for r in rows if r['phase'] == 'realign']
    ins = [r for r in rows if r['phase'] == 'insert']
    depth = [-r['priv']['tube_rel_hole_mm'][2] for r in jam]
    slip0 = rows[0]['priv']['slip_mm']
    # first step where in-hand slip grew by > 1 mm since the jam phase began (privileged label of a stall)
    stall = next((i for i, r in enumerate(jam) if r['priv']['slip_mm'] - jam[0]['priv']['slip_mm'] > 1.0), None)
    def feat(r):
        return {p[0]: (r[p]['shear_rel_px'], r[p]['press_mm']) for p in ('left_tactile', 'right_tactile')}
    out = {'seed': seed, 'success': res['check_success'], 'noise_mm': noise.tolist(),
           'jam_max_depth_mm': round(max(depth), 1), 'jam_end_slip_mm': jam[-1]['priv']['slip_mm'], 'slip_start_mm': slip0}
    if stall is not None:
        r = jam[stall]
        out.update(stall_depth_mm=round(-r['priv']['tube_rel_hole_mm'][2], 1),
                   stall_lateral_mm=r['priv']['tube_rel_hole_mm'][:2], stall_feat=feat(r),
                   pre_stall_feat=feat(jam[max(0, stall - 30)]))
    if re_:
        a, b = re_[0]['ee_mm'], re_[-1]['ee_mm']
        out['realign_dEE_mm'] = [round(b[i] - a[i], 2) for i in range(3)]
    if ins:
        out['insert_end_depth_mm'] = round(-ins[-1]['priv']['tube_rel_hole_mm'][2], 1)
        out['final_slip_mm'] = ins[-1]['priv']['slip_mm']
        out['insert_max_right_shear_x'] = round(max(abs(r['right_tactile']['shear_rel_px'][0]) for r in ins), 2)
    out['jam_max_right_shear_x'] = round(max(abs(r['right_tactile']['shear_rel_px'][0]) for r in jam), 2)
    print(json.dumps(out))
