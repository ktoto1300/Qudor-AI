# Qudor AI - Reproducibility Manifest

This document records the exact runtime environments, dependency pinning, and verification procedures required to reproduce the execution, training, and testing environments of **Qudor AI** (`qudor-ai==2.1.0`).

---

## 1. Environment Specifications

### A. Local Development Environment (Windows CPU)

| Component | Specification |
| :--- | :--- |
| **Operating System** | `Windows 11` (Build 10.0.26200 / SP0) |
| **Architecture** | `AMD64` (x86_64) |
| **Python Version** | `3.14.3` (tags/v3.14.3:323c59a, MSC v.1944 64-bit AMD64) |
| **PyTorch Version** | `2.11.0+cpu` (MKL & OpenMP enabled) |
| **NumPy Version** | `2.4.6` |
| **Pytest Version** | `9.1.1` |
| **Ruff Version** | `0.16.3` |

### B. Dedicated GPU Server Training Environment (Linux CUDA)

| Component | Specification |
| :--- | :--- |
| **Operating System** | `Ubuntu 22.04 LTS` (Linux 5.15.0+ x86_64) |
| **GPU Hardware** | `NVIDIA GeForce RTX 3060 Ti` (8 GB VRAM, Compute Capability 8.6) |
| **NVIDIA Driver** | `>= 550.142` |
| **CUDA Version** | `12.4` |
| **Python Version** | `3.10.x` (CPython Linux x86_64) |
| **PyTorch Version** | `2.5.1+cu124` (via official PyTorch wheel repository) |
| **NumPy Version** | `1.26.4` |
| **Pytest Version** | `8.3.4` |
| **Ruff Version** | `0.8.4` |

---

## 2. Locked Dependencies

- **Local CPU Development**: Locked in [`requirements-lock.txt`](requirements-lock.txt) and machine-readable in [`reproducibility.json`](reproducibility.json).
- **GPU Server Training**: Specified in [`requirements-server-cuda.txt`](requirements-server-cuda.txt). Machine-specific deployment scripts are kept outside the public repository.

### Local CPU Runtime & Development
- `numpy==2.4.6`
- `torch==2.11.0+cpu` (or `torch==2.11.0` on Linux/macOS)
- `pytest==9.1.1`
- `ruff==0.16.3`
- `setuptools==81.0.0`
- Transitive packages: `filelock==3.29.0`, `fsspec==2026.4.0`, `iniconfig==2.3.0`, `Jinja2==3.1.6`, `MarkupSafe==3.0.3`, `mpmath==1.3.0`, `networkx==3.6.1`, `packaging==26.2`, `pluggy==1.6.0`, `sympy==1.14.0`, `typing_extensions==4.15.0`, `colorama==0.4.6`.

### GPU Server Runtime
- `torch==2.5.1+cu124`
- `numpy==1.26.4`
- `pytest==8.3.4`
- `ruff==0.8.4`
- Transitive packages: `filelock>=3.13.0`, `fsspec>=2024.2.0`, `jinja2>=3.1.2`, `markupsafe>=2.1.3`, `mpmath>=1.3.0`, `networkx>=3.2.1`, `sympy>=1.12`, `typing_extensions>=4.8.0`.

---

## 3. Environment Reproduction Procedure

### Local Development Setup (Windows / CPU)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install locked dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

# 3. Install package in editable mode
python -m pip install --no-deps -e .
```

### GPU Server Setup (Ubuntu / CUDA)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install locked CUDA dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-server-cuda.txt

# 3. Install package in editable mode
python -m pip install --no-deps -e .

# 4. Verify CUDA runtime
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 4. Verification Suite

Run the standard validation commands across environments:

```bash
# 1. Bytecode compilation check across all source trees
python -m compileall -q .

# 2. Strict linting for syntax, errors, and bug risks
python -m ruff check . --select F,E9,B

# 3. Fast unit tests only
python -m pytest -m "not integration" -q

# 4. Full test suite (including integration tests)
python -m pytest -q
```
