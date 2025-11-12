# --------------------------------------------------------------
# Phase 5–6 Container for Model Unlearning, Drift, and Analysis
# Includes Jupyter Lab for notebook-based workflows.
# --------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get clean && apt-get update -o Acquire::Retries=3 -o Acquire::http::No-Cache=True && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git wget curl python3 python3-pip python3-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace
COPY . /workspace/

# Python dependencies (GPU-aware PyTorch build)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers sentence-transformers scikit-learn seaborn matplotlib pandas tqdm \
               jupyterlab notebook ipywidgets \
               numba h5py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV JUPYTER_TOKEN="tml"

# Expose port for Jupyter
EXPOSE 8888

# Default command: Start Jupyter Lab
CMD ["bash", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=$JUPYTER_TOKEN"]
