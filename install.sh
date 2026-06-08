#!/usr/bin/env bash
#
# install.sh — automated build pipeline for DGX Spark Qwen3.5-35B-A3B optimized.
#
# Phases:
#   0. Download Intel/Qwen3.5-35B-A3B-int4-AutoRound (~18 GB)
#   1. (optional, --hybrid) Download Qwen/Qwen3.5-35B-A3B-FP8 (~35 GB)
#   2. (optional, --hybrid) Build hybrid INT4+FP8 checkpoint (~15-20 min, +1.8% perf)
#   3. (optional, --hybrid) Add MTP weights back to hybrid checkpoint
#   4. Ensure vllm-sm121 base image exists (clones eugr/spark-vllm-docker if needed)
#   5. Build vllm-qwen35b-v2 final image (INT8 LM Head patch + hybrid dispatch)
#
# Flags:
#   --hybrid      Build hybrid INT4+FP8 checkpoint (adds +1.8% speed, ~40 GB extra disk, ~20 min)
#                 Skip this unless you want the last few tok/s — Phase 3 config
#                 (INT4 + MTP + INT8 LM Head) gives 98% of the total benefit.
#   --launch      After build, auto-launch the container.
#   --no-launch   Never launch. Useful for unattended runs.
#   --no-cache    Wipe existing vllm-qwen35b-v2 image and BuildKit cache, rebuild from scratch.
#   -h | --help   Print this help and exit.
#
# Sudo: this script never invokes sudo. If a prerequisite is missing it prints
# the exact command to run and exits non-zero.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_VLLM_DIR="${PROJECT_DIR}/spark-vllm-docker"
HYBRID_DIR="${HOME}/models/qwen35b-hybrid-int4fp8"
SPARK_VLLM_PIN="49d6d9fefd7cd05e63af8b28e4b514e9d30d249f"

# PyTorch nightly: resolved at runtime (see resolve_torch_nightly). Nightly wheels
# are garbage-collected after ~2 weeks, so any hardcoded date rots. We anchor on
# torchvision (the package that pins an exact torch dev-date in its metadata) and
# derive a mutually-consistent torch/torchaudio set — the same thing uv would pick
# unpinned, but captured so both Docker stages build against identical wheels.
# Override the torchvision date by exporting TORCH_NIGHTLY_DATE=YYYYMMDD.
PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cu130"
TORCH_PLATFORM_TAG="cp312-cp312-manylinux_2_28_aarch64"

# ── Flags ─────────────────────────────────────────────────────────────────────
BUILD_HYBRID=0
LAUNCH_MODE="prompt"   # prompt | yes | no
NO_CACHE=0

for arg in "$@"; do
    case "$arg" in
        --hybrid)     BUILD_HYBRID=1 ;;
        --launch)     LAUNCH_MODE="yes" ;;
        --no-launch)  LAUNCH_MODE="no" ;;
        --no-cache)   NO_CACHE=1 ;;
        -h|--help)
            sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown flag: $arg (use --help)" >&2; exit 2 ;;
    esac
done

# ── Pretty output ─────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[1;33m'
    C_BLU=$'\033[0;34m'; C_CYN=$'\033[0;36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_CYN=""; C_DIM=""; C_OFF=""
fi

log()  { echo "${C_BLU}[install]${C_OFF} $*"; }
note() { echo "${C_DIM}          $*${C_OFF}"; }
ok()   { echo "${C_GRN}[ ok ]${C_OFF}    $*"; }
warn() { echo "${C_YEL}[warn]${C_OFF}    $*"; }
err()  { echo "${C_RED}[err ]${C_OFF}    $*" >&2; }
abort(){ err "$1"; exit 1; }

STEP_NUM=0
step() {
    STEP_NUM=$((STEP_NUM + 1))
    echo
    log "${C_CYN}▶ [${STEP_NUM}] $1${C_OFF}"
    if [ -n "${2:-}" ]; then
        note "$2"
    fi
}

# ── PyTorch nightly resolution ────────────────────────────────────────────────
# nightly_dates <pkg> — YYYYMMDD dates on the cu130 index for pkg (aarch64/cp312)
nightly_dates() {
    curl -fsSL "${PYTORCH_NIGHTLY_INDEX}/$1/" 2>/dev/null \
        | grep -oE "$1-[0-9.]+\.dev[0-9]{8}\+cu130-${TORCH_PLATFORM_TAG}\.whl" \
        | grep -oE '[0-9]{8}' | sort -u
}

# nightly_version <pkg> <date> — full version string, e.g. 2.13.0.dev20260607+cu130
nightly_version() {
    curl -fsSL "${PYTORCH_NIGHTLY_INDEX}/$1/" 2>/dev/null \
        | grep -oE "$1-[0-9.]+\.dev$2\+cu130-${TORCH_PLATFORM_TAG}\.whl" \
        | head -n1 | sed -E "s/^$1-(.+)-${TORCH_PLATFORM_TAG}\.whl\$/\1/"
}

# torchvision_torch_req <date> — the exact torch version torchvision[date] pins in
# its wheel metadata (e.g. 2.13.0.dev20260603), or empty. torchvision nightlies are
# built against a *specific* torch nightly that is often a few days older than their
# own date, so we must read it rather than assume same-day.
torchvision_torch_req() {
    local href
    href=$(curl -fsSL "${PYTORCH_NIGHTLY_INDEX}/torchvision/" 2>/dev/null \
        | grep -oE "https://[^\"]*torchvision-[0-9.]+\.dev$1%2Bcu130-${TORCH_PLATFORM_TAG}\.whl" \
        | head -n1)
    [ -n "${href}" ] || return 0
    curl -fsSL --retry 2 "${href}.metadata" 2>/dev/null \
        | grep -iE '^Requires-Dist: *torch *[(]?==' \
        | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]{8}'
}

# resolve_torch_nightly — set TORCH_VERSION/TORCHVISION_VERSION/TORCHAUDIO_VERSION to
# a mutually-consistent set. Anchors on torchvision (newest available, or a forced
# date), reads the exact torch it requires from metadata, and aligns torchaudio to
# torch's date for ABI consistency. Walks back to older torchvision dates if the
# required torch wheel is no longer on the index. Reading versions off the index
# also survives base-version bumps (e.g. 2.12.0 → 2.13.0).
resolve_torch_nightly() {
    local dates d vver treq tver tdate aver adate
    if [ -n "${TORCH_NIGHTLY_DATE:-}" ]; then
        dates="${TORCH_NIGHTLY_DATE}"
        note "PyTorch nightly: anchoring on pinned torchvision date ${TORCH_NIGHTLY_DATE}"
    else
        dates=$(nightly_dates torchvision | sort -r)   # newest first
    fi

    for d in ${dates}; do
        vver=$(nightly_version torchvision "${d}" || true)
        [ -n "${vver}" ] || continue

        treq=$(torchvision_torch_req "${d}" || true)
        if [ -n "${treq}" ]; then
            tdate=$(echo "${treq}" | grep -oE '[0-9]{8}')
        else
            # metadata unreadable (e.g. older scheme with no torch pin): assume same-day
            tdate="${d}"
        fi

        tver=$(nightly_version torch "${tdate}" || true)
        [ -n "${tver}" ] || continue   # required torch wheel gone; try an older torchvision

        # torchaudio: prefer torch's date (ABI-aligned), else latest available <= d
        aver=$(nightly_version torchaudio "${tdate}" || true)
        if [ -z "${aver}" ]; then
            adate=$(nightly_dates torchaudio | awk -v hi="${d}" '$1<=hi' | tail -n1)
            [ -n "${adate}" ] && aver=$(nightly_version torchaudio "${adate}" || true)
        fi
        [ -n "${aver}" ] || continue

        TORCH_VERSION="${tver}"
        TORCHVISION_VERSION="${vver}"
        TORCHAUDIO_VERSION="${aver}"
        note "PyTorch nightly resolved (torchvision-anchored):"
        note "  torch==${TORCH_VERSION}"
        note "  torchvision==${TORCHVISION_VERSION}"
        note "  torchaudio==${TORCHAUDIO_VERSION}"
        return 0
    done

    if [ -n "${TORCH_NIGHTLY_DATE:-}" ]; then
        abort "Could not resolve a consistent PyTorch nightly set for torchvision date ${TORCH_NIGHTLY_DATE} (${TORCH_PLATFORM_TAG}). Unset TORCH_NIGHTLY_DATE to auto-detect, or pick a date from ${PYTORCH_NIGHTLY_INDEX}/torchvision/."
    fi
    abort "Could not reach/resolve the PyTorch nightly index at ${PYTORCH_NIGHTLY_INDEX}. Check connectivity, or pin a known-good date with TORCH_NIGHTLY_DATE=YYYYMMDD."
}

# ── Prerequisites ─────────────────────────────────────────────────────────────
step "Checking prerequisites"

missing=()
check() {
    local label="$1" cmd="$2" fix="$3"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  ${C_GRN}✓${C_OFF} ${label}"
    else
        echo "  ${C_RED}✗${C_OFF} ${label}   ${C_DIM}— missing${C_OFF}"
        missing+=("${label}"$'\t'"${fix}")
    fi
}

check "python3"              "command -v python3"           "sudo apt install -y python3"
check "python3-venv"         "python3 -c 'import venv'"     "sudo apt install -y python3-venv python3-pip"
check "git"                  "command -v git"               "sudo apt install -y git"
check "curl"                 "command -v curl"              "sudo apt install -y curl"
check "docker"               "command -v docker"            "https://docs.docker.com/engine/install/ubuntu/"
check "docker no-sudo"       "docker info"                  "sudo usermod -aG docker \$USER && newgrp docker"

# Disk check
need_gb=$([ "$BUILD_HYBRID" = "1" ] && echo 80 || echo 40)
free_gb=$(df -BG "${HOME}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
free_gb=${free_gb:-0}
if [ "$free_gb" -ge "$need_gb" ]; then
    echo "  ${C_GRN}✓${C_OFF} free disk ≥ ${need_gb} GB (have ${free_gb} GB)"
else
    warn "only ${free_gb} GB free in \$HOME, need ~${need_gb} GB for this config"
fi

if [ "${#missing[@]}" -gt 0 ]; then
    echo
    err "${#missing[@]} prerequisite(s) missing:"
    for item in "${missing[@]}"; do
        what="${item%%$'\t'*}"; fix="${item#*$'\t'}"
        echo "  ${C_YEL}•${C_OFF} ${what}"
        echo "    ${C_CYN}${fix}${C_OFF}"
    done
    exit 1
fi

# ── Python venv + host deps ──────────────────────────────────────────────────
step "Python venv + host-side dependencies"

cd "${PROJECT_DIR}"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q torch numpy safetensors huggingface_hub

# ── Phase 0: download Intel INT4 ──────────────────────────────────────────────
step "Phase 0 — Downloading Intel/Qwen3.5-35B-A3B-int4-AutoRound" \
     "~18 GB, first time may take 10-20 min; cached: instant"

hf download Intel/Qwen3.5-35B-A3B-int4-AutoRound
INTEL_DIR=$(hf download Intel/Qwen3.5-35B-A3B-int4-AutoRound --quiet)
[ -d "$INTEL_DIR" ] || abort "INTEL_DIR not found: ${INTEL_DIR}"
note "INTEL_DIR=${INTEL_DIR}"

# ── Phase 1+2+3: hybrid checkpoint (optional) ────────────────────────────────
if [ "$BUILD_HYBRID" = "1" ]; then
    step "Phase 1 — Downloading Qwen/Qwen3.5-35B-A3B-FP8" \
         "~35 GB, needed only for hybrid INT4+FP8 build"
    hf download Qwen/Qwen3.5-35B-A3B-FP8 >/dev/null

    # Check if hybrid checkpoint already exists
    if [ -f "${HYBRID_DIR}/model_extra_tensors.safetensors" ] \
        && [ -f "${HYBRID_DIR}/model.safetensors.index.json" ]; then
        step "Phase 2+3 — Hybrid checkpoint already exists, skipping"
        note "existing: ${HYBRID_DIR}"
    else
        step "Phase 2 — Building hybrid INT4+FP8 checkpoint" \
             "~15-20 min, output ~21 GB at ${HYBRID_DIR}"
        python3 "${PROJECT_DIR}/patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py" \
            --gptq-dir "${INTEL_DIR}" \
            --fp8-repo Qwen/Qwen3.5-35B-A3B-FP8 \
            --output "${HYBRID_DIR}" \
            --force

        step "Phase 3 — Adding MTP weights to hybrid checkpoint" \
             "restores 2329 MTP tensors that the hybrid build strips; critical for speculative decoding"
        python3 "${PROJECT_DIR}/patches/02-mtp-speculative/add-mtp-weights.py" \
            --source "${INTEL_DIR}" \
            --target "${HYBRID_DIR}"
    fi

    MODEL_SERVE_PATH="/models/qwen35b-hybrid-int4fp8"
    MODEL_MOUNT_SRC="$(dirname "${HYBRID_DIR}")"
else
    note "skipping hybrid build (use --hybrid to enable, adds ~2% speed)"
    MODEL_SERVE_PATH="Intel/Qwen3.5-35B-A3B-int4-AutoRound"
    MODEL_MOUNT_SRC=""  # no custom mount needed, HF cache is enough
fi

# ── --no-cache: wipe image and BuildKit cache ────────────────────────────────
if [ "$NO_CACHE" = "1" ]; then
    log "${C_YEL}--no-cache: removing existing image and pruning BuildKit${C_OFF}"
    docker rmi -f vllm-qwen35b-v2:latest 2>/dev/null || true
    docker builder prune -af >/dev/null 2>&1 || true
fi

# ── Phase 4: vllm-sm121 base image ────────────────────────────────────────────
if docker image inspect vllm-sm121:latest >/dev/null 2>&1; then
    step "Phase 4 — vllm-sm121 base image already exists, skipping"
    note "delete with 'docker rmi vllm-sm121' to rebuild, or pass --no-cache"
else
    step "Phase 4 — Building vllm-sm121 base image for SM121" \
         "first build: ~30-60 min; cached: ~3 min"

    if [ ! -d "${SPARK_VLLM_DIR}/.git" ]; then
        note "cloning eugr/spark-vllm-docker into ${SPARK_VLLM_DIR}"
        git clone https://github.com/eugr/spark-vllm-docker.git "${SPARK_VLLM_DIR}"
    else
        note "spark-vllm-docker already cloned, refreshing"
        git -C "${SPARK_VLLM_DIR}" fetch --quiet origin
    fi

    git -C "${SPARK_VLLM_DIR}" -c advice.detachedHead=false checkout --force "${SPARK_VLLM_PIN}"

    # Strip temporary PR patch blocks (see albond's install.sh for rationale)
    sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/&& rm pr35568.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"
    sed -i '/# TEMPORARY PATCH for broken compilation/,/&& rm pr38919.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"

    # Pin PyTorch nightly in both stages for ABI consistency
    resolve_torch_nightly
    sed -i "s|uv pip install torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/nightly/cu130|uv pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} triton --index-url https://download.pytorch.org/whl/nightly/cu130|g" \
        "${SPARK_VLLM_DIR}/Dockerfile"

    # Suppress CUTLASS×CUDA13 deprecation spam
    if ! grep -q 'NVCC_APPEND_FLAGS' "${SPARK_VLLM_DIR}/Dockerfile"; then
        sed -i '/^ENV TORCH_CUDA_ARCH_LIST=/a ENV NVCC_APPEND_FLAGS="-Xcompiler=-Wno-deprecated-declarations -diag-suppress=20012 -diag-suppress=20013 -diag-suppress=20014 -diag-suppress=20015"' \
            "${SPARK_VLLM_DIR}/Dockerfile"
    fi

    # Keep the cu130 torch nightly from being silently clobbered by a CPU wheel.
    # The Dockerfile installs torch==…dev…+cu130 first, but later `uv pip install`
    # invocations in the same stage (the vLLM/flashinfer wheels — flashinfer_python
    # declares a bare `torch` dep — plus ray/fastsafetensors) re-resolve torch. uv's
    # default prerelease policy (if-necessary-or-explicit) then refuses the installed
    # .dev nightly (not explicitly requested in *that* invocation) and pulls stable
    # torch==2.10.0 from PyPI, which on aarch64 is CPU-only → no libtorch_cuda.so →
    # `import vllm._C` fails at runtime. Allowing prereleases makes uv keep the nightly.
    if ! grep -q 'UV_PRERELEASE' "${SPARK_VLLM_DIR}/Dockerfile"; then
        sed -i -E '/^FROM .* AS (vllm-builder|runner)$/a ENV UV_PRERELEASE=allow' \
            "${SPARK_VLLM_DIR}/Dockerfile"
    fi

    (
        cd "${SPARK_VLLM_DIR}"
        ./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5
    )
    docker tag vllm-sm121:latest vllm-node-tf5:latest

    docker image inspect vllm-sm121:latest >/dev/null 2>&1 \
        || abort "vllm-sm121:latest not built"
fi

# ── Phase 5: build final image ────────────────────────────────────────────────
if docker image inspect vllm-qwen35b-v2:latest >/dev/null 2>&1 && [ "$NO_CACHE" = "0" ]; then
    step "Phase 5 — vllm-qwen35b-v2 image already exists, skipping"
    note "delete with 'docker rmi vllm-qwen35b-v2' to rebuild, or pass --no-cache"
else
    step "Phase 5 — Building vllm-qwen35b-v2 final image" \
         "thin layer over vllm-sm121: INT8 LM Head patch + hybrid INT4+FP8 dispatch"
    cd "${PROJECT_DIR}"
    docker build -t vllm-qwen35b-v2 -f docker/Dockerfile.v2 .
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo
ok "${C_GRN}All build steps complete${C_OFF}"
echo
log "Images:"
docker images vllm-sm121     --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' | grep -v '^$' || true
docker images vllm-qwen35b-v2 --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' | grep -v '^$' || true
echo

# ── Launch (prompt or auto) ──────────────────────────────────────────────────
CHAT_TEMPLATE_SRC="${PROJECT_DIR}/configs/chat_template.jinja"

build_launch_cmd() {
    local mount_model_arg=""
    if [ -n "${MODEL_MOUNT_SRC}" ]; then
        mount_model_arg="-v ${MODEL_MOUNT_SRC}:/models"
    fi

    cat <<EOF
docker run -d --name vllm-qwen35b \\
    --gpus all --net=host --ipc=host \\
    -v \${HOME}/.cache/huggingface:/root/.cache/huggingface \\
    -v ${CHAT_TEMPLATE_SRC}:/opt/unsloth.jinja:ro \\
    ${mount_model_arg} \\
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \\
    vllm-qwen35b-v2 \\
    serve ${MODEL_SERVE_PATH} \\
    --served-model-name qwen --port 8000 --host 0.0.0.0 \\
    --max-model-len 262144 --max-num-batched-tokens 16384 \\
    --gpu-memory-utilization 0.90 \\
    --reasoning-parser qwen3 \\
    --attention-backend FLASHINFER \\
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \\
    --load-format fastsafetensors --trust-remote-code \\
    --chat-template /opt/unsloth.jinja \\
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \\
    -tp 1
EOF
}

print_launch() {
    cat <<EOF
${C_CYN}To launch manually:${C_OFF}

$(build_launch_cmd)

Wait ~3-4 min for model load + warmup, then:
    curl http://127.0.0.1:8000/health
    curl http://127.0.0.1:8000/v1/models
EOF
}

do_launch() {
    log "Launching vllm-qwen35b..."
    if docker ps -a --format '{{.Names}}' | grep -qx vllm-qwen35b; then
        warn "container 'vllm-qwen35b' exists — removing"
        docker rm -f vllm-qwen35b >/dev/null
    fi
    eval "$(build_launch_cmd)" || abort "docker run failed"
    ok "container started. Use 'docker logs -f vllm-qwen35b' to watch startup."
    note "endpoint will be: http://127.0.0.1:8000/v1"
    note "stop: docker stop vllm-qwen35b"
    note "benchmark: ./bench_qwen35b.sh v2"
}

case "$LAUNCH_MODE" in
    yes)
        echo; do_launch ;;
    no)
        echo; print_launch ;;
    prompt)
        echo
        if [ ! -t 0 ]; then
            note "non-interactive shell — skipping launch prompt"
            print_launch
        else
            read -r -p "${C_CYN}Launch the container now? [y/N] ${C_OFF}" reply
            if [[ "${reply}" =~ ^[Yy]$ ]]; then
                do_launch
            else
                print_launch
            fi
        fi
        ;;
esac
