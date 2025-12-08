FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 基础依赖
RUN apt-get update && apt-get install -y \
    wget git build-essential g++ gcc \
    libgl1-mesa-glx libglib2.0-0 \
    openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev

# === 安装 Anaconda（非 Miniconda！）===
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh
ENV PATH=/opt/conda/bin:$PATH

# 创建环境（可选）
RUN conda create -y -n bev python=3.8
RUN echo "source activate bev" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=bev
ENV PATH=/opt/conda/envs/bev/bin:$PATH

# 安装 PyTorch
RUN conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch

# 使用国内 pip 源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 开启 pip 缓存目录
ENV PIP_CACHE_DIR=/root/.cache/pip


RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install mmcv==1.4.0 
RUN pip install mmcv-full==1.4.0 
RUN pip install mmdet==2.20.0
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
# # Python packages
# RUN pip install Pillow==8.4.0
# RUN pip install tqdm torchpack \
#     mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0 \
#     nuscenes-devkit mpi4py==3.0.3 numba==0.48.0
