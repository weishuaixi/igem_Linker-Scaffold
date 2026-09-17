# Install the model without RNAfold

[Back to README](../README.md)

Run commands from the repository root.


The tested server setup used Linux, Python 3.10, an RTX 5090 with 32 GB VRAM, and PyTorch 2.7.1 built for CUDA 12.8. An installed CUDA 12.6 build with no `sm_120` support previously failed on this GPU despite reporting CUDA availability. Use the explicit CUDA 12.8 wheel build; see [PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/#v271).

Run from the project directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools>=69' wheel
python -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-server.txt -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); print(y.shape)'
```

The actual GPU operation is the compatibility check; `torch.cuda.is_available()` alone is insufficient. Initial dependency installation requires network access or a separately prepared wheel cache. The dependency constraints are not a complete platform-independent lockfile; save the installed-environment record for each run.

