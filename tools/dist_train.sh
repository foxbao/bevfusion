#!/usr/bin/env bash
# 用法:
# 单机:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh 3 configs/...yaml --run-dir output/lidar_result/ [--auto-resume]
# 多机:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh 4 configs/...yaml ...

GPUS=$1
CONFIG=$2
shift 2  # 剩余参数原样传给 train.py

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29508}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"
export OMP_NUM_THREADS=$OMP_NUM_THREADS

# ---------- 自动 resume 检测 ----------
RUN_DIR=""
RESUME_ARG=""

# 提取 --run-dir 参数值
for arg in "$@"; do
    if [[ "$arg" == --run-dir ]]; then
        next_is_run_dir=1
    elif [[ $next_is_run_dir == 1 ]]; then
        RUN_DIR="$arg"
        next_is_run_dir=0
    fi
done

# 如果存在 --auto-resume 且有 run_dir，则自动找最新 checkpoint
if [[ " $@ " =~ " --auto-resume " ]] && [[ -n "$RUN_DIR" ]] && [[ -d "$RUN_DIR" ]]; then
    # 优先 latest.pth
    if [[ -f "$RUN_DIR/latest.pth" ]]; then
        CKPT="$RUN_DIR/latest.pth"
    else
        # 找最新的 epoch_*.pth
        CKPT=$(ls -t "$RUN_DIR"/epoch_*.pth 2>/dev/null | head -n 1)
    fi
    if [[ -n "$CKPT" ]]; then
        echo "🔄 Auto-resume: Resuming from $CKPT"
        RESUME_ARG="--resume-from $CKPT"
    fi
fi
# --------------------------------------

$PYTHON_BIN -m torch.distributed.run \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$PORT \
  "$(dirname "$0")/train_modified.py" \
  "$CONFIG" \
  --launcher pytorch "$@" $RESUME_ARG
