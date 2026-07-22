#!/bin/bash
set -euo pipefail

REPO_URL="${1:-https://github.com/tanzimul3islam/AdaGAD-HNC.git}"
PAT="${2:-}"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  wget \
  git \
  ca-certificates \
  software-properties-common \
  ubuntu-drivers-common

ubuntu-drivers autoinstall

curl -fsSL https://developer.download.nvidia.com/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
apt-get update
apt-get install -y --no-install-recommends cuda-toolkit-12-2

echo "export PATH=/usr/local/cuda/bin:${PATH}" >> /home/azureuser/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" >> /home/azureuser/.bashrc
echo "export NVIDIA_VISIBLE_DEVICES=all" >> /home/azureuser/.bashrc
echo "export NVIDIA_DRIVER_CAPABILITIES=compute,utility" >> /home/azureuser/.bashrc

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p /home/azureuser/miniconda3
rm /tmp/miniconda.sh

chown -R azureuser:azureuser /home/azureuser/miniconda3

echo 'export PATH=/home/azureuser/miniconda3/bin:$PATH' >> /home/azureuser/.bashrc

if [ -n "$PAT" ]; then
  REPO_URL="https://${PAT}@github.com/tanzimul3islam/AdaGAD-HNC.git"
fi

su - azureuser -c "git clone ${REPO_URL} /home/azureuser/AdaGAD-HNC || true"
