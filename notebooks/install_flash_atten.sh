which python
python -c "import sys; print(sys.executable)"

python -m pip uninstall -y flash-attn flash_attn || true
python -m pip cache purge
python - <<'PY'
import site, glob, os, shutil
for d in set(site.getsitepackages()+[site.getusersitepackages()]):
    for pat in ("flash_attn*", "flash-attn*"):
        for p in glob.glob(os.path.join(d, pat)):
            print("REMOVE", p)
            if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p): os.remove(p)
print("DONE")
PY

conda install -y -c nvidia cuda-nvcc=12.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V
python -m pip install -U pip setuptools wheel ninja packaging

export PIP_NO_BINARY=flash-attn
export PIP_ONLY_BINARY=":none:"
export FLASH_ATTENTION_FORCE_BUILD=1

python -m pip install --no-binary :all: --no-build-isolation "flash-attn==2.7.4.post1"

python - <<'PY'
import flash_attn, importlib.util, subprocess
print("flash_attn version:", getattr(flash_attn, "__version__", "unknown"))
spec = importlib.util.find_spec("flash_attn_2_cuda")
print("flash_attn_2_cuda path:", spec.origin if spec else None)
out = subprocess.check_output(["strings", spec.origin], text=True, errors="ignore")
print("GLIBC symbols:", sorted(set(s for s in out.splitlines() if s.startswith("GLIBC_"))))
PY
