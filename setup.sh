#! /bin/bash

# Linux x86_64 (Ubuntu 18.04/20.04/22.04)
# If the arch is different, please refer to the official website: https://developer.nvidia.com/cuda-toolkit-archive
# And also this is environment for colab, you can aslo change the version of cuda and torch if needed.
# colab environment: cuda:12.1, torch:2.3.0+cu121
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo chmod +x cuda_12.1.0_530.30.02_linux.run 
sudo sh cuda_12.1.0_530.30.02_linux.run
pip install torch, numpy, pandas

