#!/bin/bash

distro='rhel7'
arch='x86_64'

# install nvidia drivers
yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
subscription-manager repos --enable=rhel-7-workstation-optional-rpms

rpm --erase gpg-pubkey-7fa2af80*

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.repo
yum clean expire-cache
yum install nvidia-driver-latest-dkms
yum install cuda
yum install cuda-drivers
ln -s /usr/lib/nvidia/libcuda.so /usr/lib/libcuda.so
ln -s /usr/lib/nvidia/libcuda.so /usr/lib64/libcuda.so

# install nvidia runtime for docker

yum install -y tar bzip2 make automake gcc gcc-c++ vim pciutils elfutils-libelf-devel libglvnd-devel iptables
distribution=$(
  . /etc/os-release
  echo $ID$VERSION_ID
) &&
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

yum-config-manager --enable libnvidia-container-experimental
yum clean expire-cache
yum install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Test nvidia-smi with the latest official CUDA image
