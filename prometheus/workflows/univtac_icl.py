"""RoboICL-style visuo-tactile ICL on the official UniVTAC benchmark.

operation=build_reference  one official HDF5 episode -> TRAIN reference bundle
operation=prepare          frozen per-seed runtimes for one condition (no simulator, no model)
operation=run              execute a prepared run with the official evaluator

Only the TRAIN/LIVE modality condition changes between runs of one comparison.
"""
import json
import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from prometheus.policy.univtac_icl import reference as reference_io
from prometheus.workflows import univtac_qwen as official

PROJECT = Path(__file__).resolve().parents[2]
POLICY_PACKAGE = 'prometheus.policy.univtac_icl'
CONDITIONS = {  # name -> (train_modality, live_tactile)
    'zero_shot': ('none', False),
    'zero_shot_tactile': ('none', True),
    'visual': ('vision', False),
    'visual_live_tactile': ('vision', True),
    'visuotactile': ('vision_tactile', True),
}


def build_reference(config):
    output = official.payload_path(config['reference_output'])
    path = reference_io.build(config['reference_source'], output, config['task'], config['reference_source_id'],
                              config['reference_metadata'],
                              config['reference_waypoint_thresholds'], config['policy']['horizon'],
                              config['policy']['limits']['finger_max_mm'],
                              jam_frames=config.get('reference_jam_frames'))
    return path


def deployment(config, runtime, seed, reference_path):
    train_modality, live_tactile = CONDITIONS[config['condition']]
    policy = dict(config['policy'], policy_name='PrometheusICL', task_name=config['task'],
                  run_id=f"{config['run_id']}-s{seed}", audit_dir=str(runtime / 'audit'),
                  condition=config['condition'], train_modality=train_modality, live_tactile=live_tactile,
                  reference=None, reference_sha256=None)
    if train_modality != 'none':
        policy.update(reference=str(reference_path), reference_sha256=reference_io.sha256(reference_path))
    return policy


def prepare(config):
    if config['condition'] not in CONDITIONS:
        raise ValueError(f'condition must be one of {sorted(CONDITIONS)}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['run_id']):
        raise ValueError('run_id must be a simple unique identifier')
    seeds = config['seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Explicit unique nonnegative seeds required')
    source = Path(config['source_root']).resolve()
    official.verify_source(source)
    task_config = (source / 'task_config' / config['task_config']).resolve()
    if not task_config.is_file():
        raise ValueError('Select an existing official task config file')
    kit_args = official.livestream_arguments(config['runtime_ports'])
    reference_path = official.payload_path(config['reference_output']) / 'reference.json'
    if CONDITIONS[config['condition']][0] != 'none':
        # Fail before creating any runtime if the bundle is missing or inconsistent.
        reference_io.load(reference_path, config['task'], config['policy']['horizon'])
    output = official.payload_path(config['output_root']) / config['run_id']
    output.mkdir(parents=True, exist_ok=False)
    bindings = {str(p): reference_io.sha256(p) for p in
                [*sorted((PROJECT / 'prometheus/policy/univtac_icl').glob('*.py')), Path(__file__).resolve(), task_config]}
    probe = output / 'probe_camera.py'
    probe.write_text(official.probe_script(output / 'camera_ready.json', config['device']))
    bindings[str(probe)] = reference_io.sha256(probe)
    jobs = []
    for seed in seeds:
        runtime = output / f'seed_{seed}'
        runtime.mkdir()
        # Isolated cwd; official source and assets stay read-only symlinks.
        for entry in source.iterdir():
            if not entry.name.startswith('.') and entry.name not in {'data', 'eval_result', 'log'}:
                (runtime / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        for directory in ('data', 'eval_result', 'log'):
            (runtime / directory).mkdir()
        deploy_path = runtime / 'deploy.yaml'
        policy = deployment(config, runtime, seed, reference_path)
        OmegaConf.save(OmegaConf.create(policy), deploy_path)
        if policy['reference'] is not None:
            bindings[policy['reference']] = policy['reference_sha256']
        entry = runtime / 'entry.py'
        argv = [str(source / 'scripts/eval_policy.py'), config['task'], str(task_config), str(deploy_path),
                '--start_seed', str(seed), '--max_seed', str(seed), '--total_num', '1',
                '--headless', '--device', config['device'], '--kit_args', kit_args]
        entry.write_text(
            'import sys, runpy, importlib\n'
            f'sys.path.insert(0, {str(PROJECT)!r})\n'
            f'sys.path.insert(0, {str(source)!r})\n'
            'import policy\n'
            f'sys.modules["policy.PrometheusICL"] = importlib.import_module({POLICY_PACKAGE!r})\n'
            f'sys.argv = {argv!r}\n'
            'runpy.run_path(sys.argv[0], run_name="__main__")\n')
        bindings[str(deploy_path)] = reference_io.sha256(deploy_path)
        bindings[str(entry)] = reference_io.sha256(entry)
        jobs.append(dict(seed=seed, cwd=str(runtime), argv=[config['python'], str(entry)], status='not_run'))
    manifest = dict(schema='univtac_icl_run_v1', status='prepared', revision=official.REVISION,
                    source_root=str(source), task=config['task'], run_id=config['run_id'],
                    condition=config['condition'], planned=len(seeds), bindings=bindings, config=config, jobs=jobs,
                    success_source='official evaluator', hardware_commands=False)
    official.dump(output / 'prepared_plan.json', manifest)
    manifest['prepared_plan_sha256'] = reference_io.sha256(output / 'prepared_plan.json')
    official.dump(output / 'manifest.json', manifest)
    return output


@hydra.main(version_base=None, config_path='../configs', config_name='univtac_icl')
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    if config['operation'] == 'build_reference':
        print(build_reference(config))
    elif config['operation'] == 'prepare':
        print(prepare(config))
    elif config['operation'] == 'run':
        manifest = json.loads((Path(config['prepared_run']) / 'manifest.json').read_text())
        if manifest.get('schema') != 'univtac_icl_run_v1':
            raise ValueError('Not a univtac_icl prepared run')
        official.run(config['prepared_run'])
    else:
        raise ValueError('operation must be build_reference, prepare or run')


if __name__ == '__main__':
    main()
