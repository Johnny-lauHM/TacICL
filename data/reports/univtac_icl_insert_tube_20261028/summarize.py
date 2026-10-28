"""Summarize insert_tube comparison runs (v5 direct control and the jam-guard groups).

A seed's outcome comes from the latest run that gave it a policy outcome (success/failed); infrastructure
reruns (<run>_infraN) only cover seeds that ended as error before any model call. The hidden pre_move
offset (official metadata random_noise, hole frame) is read after the fact for analysis only.
Prints a markdown table and writes summary.json next to this file.
"""
import glob
import json
import statistics
from pathlib import Path

RUNS = Path('/home/nus/workspace/PrometheusV4/data/runs/univtac_icl')
GROUPS = {
    'v5 direct / zero_shot': 'insert_tube_v5_zero_shot_20261028',
    'v5 direct / visual': 'insert_tube_v5_visual_20261028',
    'v5 direct / visuotactile': 'insert_tube_v5_visuotactile_20261028',
    'G1 guard+hint / visuotactile': 'insert_tube_guardG1_visuotactile_20261028',
    'G2 guard / visuotactile': 'insert_tube_guardG2_visuotactile_20261028',
    'G3 guard+hint / visual': 'insert_tube_guardG3_visual_20261028',
    # 30-seed comparison (seeds 1000001-1000030)
    'S30 v2est guard+estimate / visuotactile': 'insert_tube_v2est_visuotactile_20261028',
    'S30 v5rep direct / visual': 'insert_tube_v5rep_visual_20261028',
    'S30 g1rep guard+direction / visuotactile': 'insert_tube_g1rep_visuotactile_20261028',
}


def episodes(base):
    seeds = {}
    for run in sorted(glob.glob(str(RUNS / base))) + sorted(glob.glob(str(RUNS / (base + '_infra*')))):
        for job in json.load(open(Path(run) / 'manifest.json'))['jobs']:
            status = job.get('status')
            if status in ('success', 'failed') or (status in ('error', 'timeout') and job['seed'] not in seeds):
                seeds[job['seed']] = (status, job['cwd'], job.get('wall_s', 0))
    return seeds


def metrics(cwd):
    events = [json.loads(l) for l in (Path(cwd) / 'audit' / 'events.jsonl').open()] if (Path(cwd) / 'audit' / 'events.jsonl').exists() else []
    executed = [e for e in events if e['event'] == 'executed']
    responses = [e for e in events if e['event'] == 'response']
    stops = [e['receipt'].get('stop_reason') for e in executed]
    meta = sorted(glob.glob(str(Path(cwd) / 'eval_result/*/*/*/*/metadata.json')))
    record = list(json.load(open(meta[-1])).values())[0] if meta else {}
    noise = record.get('random_noise', [None, None])[:2]
    return {'calls': len(executed),
            'tokens': statistics.mean(r['usage']['completion_tokens'] for r in responses) if responses else 0,
            'official_early_stop': 'official_early_stop' in stops,
            'jams': sum(s == 's1_jam' for s in stops),
            'noise_mm': [round(v * 1000, 1) for v in noise] if noise[0] is not None else None,
            'inhand_bias_mm': round(record['inhand_bias'] * 1000, 1) if isinstance(record.get('inhand_bias'), (int, float)) else None,
            'end': next((e['reason'] for e in reversed(events) if e['event'] == 'episode_end'), stops[-1] if stops else None)}


def main():
    out = {}
    lines = ['| group | success | failed | infra error | early stop | success y>0 / y<0 | mean calls | jam stops | mean tokens/call |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, base in GROUPS.items():
        seeds = episodes(base)
        if not seeds:
            continue
        rows = {seed: dict(status=s, wall_s=round(w), **metrics(cwd)) for seed, (s, cwd, w) in sorted(seeds.items())}
        out[name] = rows
        done = [r for r in rows.values() if r['status'] in ('success', 'failed')]
        pos = [r for r in done if r['noise_mm'] and r['noise_mm'][1] > 0]
        neg = [r for r in done if r['noise_mm'] and r['noise_mm'][1] < 0]
        lines.append('| {} | {} | {} | {} | {} | {}/{} · {}/{} | {:.1f} | {} | {:.0f} |'.format(
            name, sum(r['status'] == 'success' for r in done), sum(r['status'] == 'failed' for r in done),
            sum(r['status'] not in ('success', 'failed') for r in rows.values()),
            sum(r['official_early_stop'] for r in done),
            sum(r['status'] == 'success' for r in pos), len(pos), sum(r['status'] == 'success' for r in neg), len(neg),
            statistics.mean(r['calls'] for r in done) if done else 0, sum(r['jams'] for r in done),
            statistics.mean(r['tokens'] for r in done) if done else 0))
    print('\n'.join(lines))
    json.dump(out, open(Path(__file__).with_name('summary.json'), 'w'), indent=1)
    for name, rows in out.items():
        print(f'\n{name}:', ' '.join(f"{seed}:{r['status'][0]}" for seed, r in rows.items()))


if __name__ == '__main__':
    main()
