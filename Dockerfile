FROM ghcr.io/pytorch/pytorch-nightly:2.9.0.dev20250803-cuda12.9-cudnn9-devel

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Blackwell support and CUDA development
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDACXX=$CUDA_HOME/bin/nvcc
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Clone diffusion-pipe with submodules
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe.git /diffusion-pipe

WORKDIR /diffusion-pipe

# Install flash-attn from official precompiled wheels first (tested and working)
RUN pip3 install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install additional optimizers
RUN pip3 install --no-cache-dir came-pytorch

# Configure accelerate defaults for diffusion-pipe
RUN mkdir -p /root/.cache/huggingface/accelerate && \
    echo '{\n\
  "compute_environment": "LOCAL_MACHINE",\n\
  "distributed_type": "NO",\n\
  "num_processes": 1,\n\
  "num_machines": 1,\n\
  "machine_rank": 0,\n\
  "dynamo_backend": "no",\n\
  "gpu_ids": "all",\n\
  "mixed_precision": "bf16",\n\
  "main_training_function": "main"\n\
}' > /root/.cache/huggingface/accelerate/default_config.json

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"] 