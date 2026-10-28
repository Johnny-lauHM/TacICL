#!/bin/bash
# Diagnostic jam-signal calibration on development seeds (never evaluation seeds 1000001-1000010).
# Usage: [PROBE=probe_multi.py] run_jam_probe.sh RUN_ID SEED...  (default PROBE=jam_probe.py)
set -u
cd /home/nus/workspace/PrometheusV4
PY=$PWD/data/envs/univtac-isaac51/bin/python
run=$1; shift
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=$PWD/data/cache/univtac_runtime TORCH_EXTENSIONS_DIR=$PWD/data/cache/univtac_torch_extensions
seeds=$(IFS=,; echo "$*")
PYTHONPATH=$PWD $PY -m prometheus.workflows.univtac_icl --config-name univtac_icl_insert_tube operation=prepare run_id=$run condition=zero_shot "seeds=[$seeds]" python=$PY > /dev/null
for s in "$@"; do
  d=data/runs/univtac_icl/$run/seed_$s
  sed -e "s#importlib.import_module('prometheus.policy.univtac_icl')#runpy.run_path('$PWD/data/reports/univtac_icl_insert_tube_20261028/${PROBE:-jam_probe.py}')#" $d/entry.py > $d/entry_probe.py
  # run_path returns a dict; wrap it as a module-like namespace for policy.PrometheusICL.Policy
  sed -i "s#sys.modules\[\"policy.PrometheusICL\"\] = runpy.run_path(\(.*\))#import types; _m = types.ModuleType('jam_probe'); _m.__dict__.update(runpy.run_path(\1)); sys.modules[\"policy.PrometheusICL\"] = _m#" $d/entry_probe.py
  (cd $d && timeout 900 $PY entry_probe.py > probe.log 2>&1)
  echo "$(date '+%F %T') seed $s exit=$? $(cat $d/audit/result.json 2>/dev/null | tr -d '\n' | cut -c1-200)"
done
