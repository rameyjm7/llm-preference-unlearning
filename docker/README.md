# LLM Preference Unlearning — Docker Environment

This directory contains the full containerization and orchestration setup for running LLM Preference Unlearning experiments on both local GPU workstations and HPC environments (DGX / Apptainer).  
It packages all dependencies required for Phases 5–6 of the project — activation-level unlearning, Fisher/saliency analysis, and re-evaluation — and provides a ready-to-use Jupyter Lab workspace.

## Contents

| File | Purpose |
|------|----------|
| **Dockerfile** | Defines the base GPU container (`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`) with PyTorch, Transformers, Jupyter Lab, and analysis libraries pre-installed. |
| **Makefile** | Unified build and run automation for Docker and Apptainer (HPC). Supports building, pushing, running, and converting to `.sif` images. |
| **docker-compose.yml** | Simple orchestration for running Jupyter Lab locally with NVIDIA GPU runtime and persistent workspace mounts. |

## Environment Overview

**Base image:** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`  
**Installed stacks:**
- Python 3 + PyTorch 2 (CUDA 12.1 build)
- Hugging Face Transformers + Sentence-Transformers
- scikit-learn, pandas, seaborn, matplotlib
- Jupyter Lab + ipywidgets
- Numba, h5py for activation analysis

**Exposed service:**  
Jupyter Lab on port `8888` with token `tml`

## Usage

### 1. Build Docker Image
Builds an x86_64 (AMD64) image.

```
make build
```

### 2. Run Locally via Docker Compose
Launches Jupyter Lab with GPU support and workspace volume mounted.

```
make run
```

Access at: [http://localhost:8888/?token=tml](http://localhost:8888/?token=tml)

Stop the container with:
```
make stop
```

## HPC / Apptainer Workflow

### 3. Build Apptainer `.sif` from Docker Hub

```
make sif
```

This pulls the latest Docker image from Docker Hub and converts it into an Apptainer-compatible `.sif` file.

### 4. Launch Jupyter Lab inside Apptainer
```
make apptainer-lab
```
Starts Jupyter Lab within the containerized GPU environment on port 8888.

### 5. Run Scripts Non-Interactively
To execute a script such as `activation_unlearning.py` directly inside Apptainer:
```
make apptainer-run
```

## Cleanup

Remove the local Docker image (optional):

```
make clean
```

## Notes

- The image is built for x86_64 (AMD64) to ensure compatibility with NVIDIA DGX and other HPC GPU nodes.
- Use `--network=host` during Docker builds if your environment has restricted DNS or proxy settings.
- The `workspace` directory is automatically mounted, ensuring notebooks and output files persist outside the container.

Maintainer:  
Jacob Ramey (https://github.com/rameyjm7)  
Last updated: November 2025
