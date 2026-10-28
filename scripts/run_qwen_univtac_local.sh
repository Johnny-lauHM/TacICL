#!/usr/bin/env bash
set -euo pipefail

# Reuse installed IQ3_S weights; no model conversion or larger quantization.
# The installer accepts int8, while the engine override selects verified FP16 KV.
# MTP drafting on (lossless at temperature 0; MotorMind serves with NEXTN speculative decoding).
exec /home/nus/.local/bin/strata launch \
  --preset univtac-iq3-s-32k --save-preset univtac-robot-32k \
  --context 32k --parallel 1 --vision gpu --gpu 0 \
  --kv int8 --set kv=fp16 \
  --mtp on --drafting mtp --set spec=4 --set suffix-draft=0 \
  --set STRATA_ROUTE_TAIL_SKIP=0 --set "vram-reserve-mib=${QWEN_VRAM_RESERVE_MIB:-16384}" \
  --access local --port 18329 --restart --no-tui --yes
