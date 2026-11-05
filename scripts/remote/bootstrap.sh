#!/usr/bin/env bash
set -euo pipefail

SKIP_TESTS=0
for a in "$@"; do
  case "$a" in
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-data) ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
ENV_NAME=utrscan
say() { printf '\n\033[1m[bootstrap] %s\033[0m\n' "$*"; }

[ "$(uname -s)" = "Linux" ] || { echo "This script targets Linux."; exit 1; }

if ! command -v conda >/dev/null 2>&1; then
  if [ -x "$HOME/miniforge3/bin/conda" ]; then
    :
  else
    say "installing Miniforge into ~/miniforge3"
    ARCH="$(uname -m)"
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
  fi
  export PATH="$HOME/miniforge3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
SOLVER=conda
command -v mamba >/dev/null 2>&1 && SOLVER=mamba

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "updating environment $ENV_NAME"
  $SOLVER env update -n "$ENV_NAME" -f environment.yml --prune
else
  say "creating environment $ENV_NAME (a few minutes)"
  $SOLVER env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  say "NVIDIA GPU found: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    say "installing PyTorch (CUDA build)"
    pip install --upgrade "torch>=2.5"
  fi
  python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("\nPyTorch cannot use the GPU (driver too old for this CUDA build?).\n"
             "The pipeline will still run, on CPU.  `nvidia-smi` shows the driver version;\n"
             "PyTorch needs a driver supporting CUDA 12.x.\n")
print("[bootstrap] torch", torch.__version__, "CUDA", torch.version.cuda, "->",
      torch.cuda.get_device_name(0))
EOF
else
  say "no NVIDIA GPU visible: installing the CPU build of PyTorch"
  python -c "import torch" 2>/dev/null || \
    pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu
fi

say "checking tools"
bowtie2-align-s --version | head -n1
bowtie2-build-s --version | head -n1
echo "cutadapt $(cutadapt --version)"
for t in prefetch fasterq-dump pigz parallel; do
  command -v "$t" >/dev/null 2>&1 || { echo "missing $t (needed by download_srrs.sh)"; exit 1; }
done
echo "sra-tools $(prefetch --version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -n1), pigz, parallel: ok"
python -c "import RNA; print('ViennaRNA', RNA.__version__)"
python -c "import numpy, scipy, sklearn, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
python -c "
from analysis.recoverability import config as C
r = C.resources(); print('[bootstrap] resources', r, '-> backend', C.backend(), 'workers', C.default_workers())"

if [ "$SKIP_TESTS" = 0 ]; then
  say "unit tests"
  python -m analysis.recoverability.tests.test_likelihood
  python -m analysis.recoverability.tests.test_nulls
  python -m analysis.recoverability.tests.test_chainfast
  python -m analysis.recoverability.tests.test_backends
fi

touch "$REPO/.bootstrap_ok"
say "done.  Next:
    conda activate $ENV_NAME
    bash download_srrs.sh                    # FASTQs + references (skips what is present)
    bash scripts/remote/run_pipeline.sh      # the whole analysis"
