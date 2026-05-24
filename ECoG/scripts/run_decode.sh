#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-}"
if [[ -z "$CFG" ]]; then
  echo "Usage: $0 configs/exp/xxx.yaml" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

shift
EXTRA_ARGS=("$@")

# Guard against accidental multi-config positional inputs.
# run_decode.sh supports exactly one config path; additional positional
# YAML paths are almost always user mistakes and can make failures hard to debug.
for arg in "${EXTRA_ARGS[@]:-}"; do
  if [[ "$arg" == *.yaml || "$arg" == *.yml ]]; then
    if [[ -f "$arg" ]]; then
      echo "[ERROR] Detected extra config-like positional argument: $arg" >&2
      echo "[ERROR] Use queue.sh for multiple configs, or pass exactly one config to run_decode.sh." >&2
      exit 2
    fi
  fi
done

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  echo "[run.sh] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
fi

EXP_NAME="$(python -c "import sys,yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get('exp_name','exp'))" "$CFG")"
TASK="$(python -c "import sys,yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get('task',''))" "$CFG")"
if [[ -z "$TASK" ]]; then
  echo "Missing task in config.yaml (key: task)" >&2
  exit 1
fi

TS="$(date +"%Y-%m-%d_%H%M%S")"
OUT_ROOT="$(
  python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])) or {}; run=(cfg.get('run',{}) or {}); print(run.get('out_root','outputs/runs'))" "$CFG"
)"
USE_RUNNING="$(
  python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])) or {}; run=(cfg.get('run',{}) or {}); print(str(run.get('use_running_dir', True)).lower())" "$CFG"
)"
RUNNING_ROOT="$(
  python -c "import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])) or {}; run=(cfg.get('run',{}) or {}); print(run.get('running_root','outputs/runs/running'))" "$CFG"
)"

if [[ "$TASK" == "train_sft" || "$TASK" == "train_tapt" || "$TASK" == "train_lora" || "$TASK" == "encoder_classify" || "$TASK" == "train_decoder_classifier" || "$TASK" == "train_ccia" || "$TASK" == "train_ccia_curriculum" ]]; then
  if [[ "$USE_RUNNING" == "true" ]]; then
    RUN_ROOT="$RUNNING_ROOT"
  else
    RUN_ROOT="$OUT_ROOT"
  fi
else
  RUN_ROOT="$OUT_ROOT"
fi

RUN_DIR="${RUN_ROOT}/${TS}_${EXP_NAME}"
mkdir -p "$RUN_DIR"

cp "$CFG" "$RUN_DIR/config.yaml"

# Optional CUDA_VISIBLE_DEVICES from config: run.cuda_visible_devices
CFG_CUDA="$(python - "$RUN_DIR/config.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
raw = (cfg.get("run", {}) or {}).get("cuda_visible_devices", "")
print("" if raw is None else str(raw))
PY
)"
if [[ -n "$CFG_CUDA" ]]; then
  CFG_CUDA_SANITIZED="$(
    python - "$CFG_CUDA" <<'PY'
import sys
raw = str(sys.argv[1] or "")
seen = set()
tokens = []
for token in raw.split(","):
    t = token.strip()
    if not t:
        continue
    if t in seen:
        continue
    seen.add(t)
    tokens.append(t)
print(",".join(tokens))
PY
  )"
  export CUDA_VISIBLE_DEVICES="$CFG_CUDA_SANITIZED"
  if [[ "$CFG_CUDA_SANITIZED" != "$CFG_CUDA" ]]; then
    echo "[run.sh] CUDA_VISIBLE_DEVICES normalized: '$CFG_CUDA' -> '$CUDA_VISIBLE_DEVICES'"
  else
    echo "[run.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from config)"
  fi
fi

N_GPUS="$(
  python - "$RUN_DIR/config.yaml" <<'PY'
import os, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
run = (cfg.get("run", {}) or {})
cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
if not cuda:
    cuda = run.get("cuda_visible_devices")

def parse_cuda_count(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len([v for v in value if str(v).strip() != ""])
    s = str(value).strip()
    if not s:
        return None
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    return len(tokens) if tokens else None

derived = parse_cuda_count(cuda)
if derived is not None and derived > 0:
    print(derived)
else:
    try:
        n = int(run.get("n_gpus", 1))
    except Exception:
        n = 1
    print(max(1, n))
PY
)"
echo "[run.sh] n_gpus=$N_GPUS (derived from run.cuda_visible_devices when set, else run.n_gpus)"

echo "[run.sh] task=$TASK run_dir=$RUN_DIR"

case "$TASK" in
  eval_decode)
    python -m phishdec.cli.eval_decode --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs.txt"
    ;;

  eval_decoder_classifier)
    python -m phishdec.cli.eval_decoder_classifier --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs.txt"
    ;;

  train_sft|train_tapt)
    if [[ "$N_GPUS" -gt 1 ]]; then
      command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found. Install torch first." >&2; exit 1; }
      echo "[run.sh] torchrun nproc_per_node=$N_GPUS"
      MASTER_PORT="$(
        python - "$RUN_DIR/config.yaml" <<'PY'
import yaml,sys,socket,random
cfg=yaml.safe_load(open(sys.argv[1])) or {}
p=int((cfg.get("run",{}) or {}).get("master_port", 29500))

def free(port):
    try:
        s=socket.socket()
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False

if free(p):
    print(p)
else:
    for _ in range(50):
        q=random.randint(20000, 45000)
        if free(q):
            print(q)
            break
    else:
        print(p)
PY
      )"

      echo "[run.sh] master_port=$MASTER_PORT"

      torchrun --master_port "$MASTER_PORT" --nproc_per_node "$N_GPUS" \
        -m phishdec.cli.train_sft --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    else
      python -m phishdec.cli.train_sft --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    fi
    ;;

  train_lora)
    if [[ "$N_GPUS" -gt 1 ]]; then
      command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found. Install torch first." >&2; exit 1; }
      echo "[run.sh] torchrun nproc_per_node=$N_GPUS"
      MASTER_PORT="$(
        python - "$RUN_DIR/config.yaml" <<'PY'
import yaml,sys,socket,random
cfg=yaml.safe_load(open(sys.argv[1])) or {}
p=int((cfg.get("run",{}) or {}).get("master_port", 29500))

def free(port):
    try:
        s=socket.socket()
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False

if free(p):
    print(p)
else:
    for _ in range(50):
        q=random.randint(20000, 45000)
        if free(q):
            print(q)
            break
    else:
        print(p)
PY
      )"

      echo "[run.sh] master_port=$MASTER_PORT"

      torchrun --master_port "$MASTER_PORT" --nproc_per_node "$N_GPUS" \
        -m phishdec.cli.train_lora --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    else
      python -m phishdec.cli.train_lora --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    fi
    ;;

  tapt_mlm)
    python -m phishdec.cli.tapt_mlm --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs.txt"
    ;;

  encoder_classify)
    python -m phishdec.cli.encoder_classify --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs.txt"
    ;;

  train_decoder_classifier)
    if [[ "$N_GPUS" -gt 1 ]]; then
      command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found. Install torch first." >&2; exit 1; }
      echo "[run.sh] torchrun nproc_per_node=$N_GPUS"
      MASTER_PORT="$(
        python - "$RUN_DIR/config.yaml" <<'PY'
import yaml,sys,socket,random
cfg=yaml.safe_load(open(sys.argv[1])) or {}
p=int((cfg.get("run",{}) or {}).get("master_port", 29500))

def free(port):
    try:
        s=socket.socket()
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False

if free(p):
    print(p)
else:
    for _ in range(50):
        q=random.randint(20000, 45000)
        if free(q):
            print(q)
            break
    else:
        print(p)
PY
      )"

      echo "[run.sh] master_port=$MASTER_PORT"

      torchrun --master_port "$MASTER_PORT" --nproc_per_node "$N_GPUS" \
        -m phishdec.cli.train_decoder_classifier --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    else
      python -m phishdec.cli.train_decoder_classifier --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    fi
    ;;

  train_ccia)
    if [[ "$N_GPUS" -gt 1 ]]; then
      command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found. Install torch first." >&2; exit 1; }
      echo "[run.sh] torchrun nproc_per_node=$N_GPUS"
      MASTER_PORT="$(
        python - "$RUN_DIR/config.yaml" <<'PY'
import yaml,sys,socket,random
cfg=yaml.safe_load(open(sys.argv[1])) or {}
p=int((cfg.get("run",{}) or {}).get("master_port", 29500))

def free(port):
    try:
        s=socket.socket()
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False

if free(p):
    print(p)
else:
    for _ in range(50):
        q=random.randint(20000, 45000)
        if free(q):
            print(q)
            break
    else:
        print(p)
PY
      )"

      echo "[run.sh] master_port=$MASTER_PORT"

      torchrun --master_port "$MASTER_PORT" --nproc_per_node "$N_GPUS" \
        -m phishdec.cli.train_ccia --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    else
      python -m phishdec.cli.train_ccia --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    fi
    ;;

  train_ccia_curriculum)
    if [[ "$N_GPUS" -gt 1 ]]; then
      command -v torchrun >/dev/null 2>&1 || { echo "torchrun not found. Install torch first." >&2; exit 1; }
      echo "[run.sh] torchrun nproc_per_node=$N_GPUS"
      MASTER_PORT="$(
        python - "$RUN_DIR/config.yaml" <<'PY'
import yaml,sys,socket,random
cfg=yaml.safe_load(open(sys.argv[1])) or {}
p=int((cfg.get("run",{}) or {}).get("master_port", 29500))

def free(port):
    try:
        s=socket.socket()
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False

if free(p):
    print(p)
else:
    for _ in range(50):
        q=random.randint(20000, 45000)
        if free(q):
            print(q)
            break
    else:
        print(p)
PY
      )"

      echo "[run.sh] master_port=$MASTER_PORT"

      torchrun --master_port "$MASTER_PORT" --nproc_per_node "$N_GPUS" \
        -m phishdec.cli.train_ccia_curriculum --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    else
      python -m phishdec.cli.train_ccia_curriculum --config "$RUN_DIR/config.yaml" --out_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/logs.txt"
    fi
    ;;

  *)
    echo "Unknown task: $TASK" >&2
    exit 1
    ;;
esac

ln -sfn "$RUN_DIR" outputs/latest
echo "[run.sh] done."
