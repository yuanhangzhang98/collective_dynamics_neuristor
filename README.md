# Collective dynamics and long-range order in thermal neuristor networks
This repository contains source code for reproducing the results in the paper *Collective dynamics and long-range order in thermal neuristor networks*. ArXiv link: [arXiv:2312.12899](https://arxiv.org/abs/2312.12899v1). 

Here is another relevant paper on thermal neuristors: [Reconfigurable Cascaded Thermal Neuristors for Neuromorphic Computing](https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.202306818)

Requirements: NumPy, SciPy, PyTorch >= 2.3.0, TorchVision. The code is tested in the following environment: Linux, NVIDIA TITAN RTX GPU with 24GB memory, CUDA 11.3, Python 3.9.13, NumPy 1.23.1, SciPy 1.9.3, PyTorch 2.3.0, TorchVision 0.18.0. 

Usage: 

`python main.py` computes the avalanche size distribution with specific parameters. Takes about 30 minutes in the test environment. 
Under the default setting, this script computes the distribution corresponding to Fig. 3(a) in the paper, saves the data to the directory `avalanches/`, and plots the snapshots of the dynamics, saving them to the directory `graphs/`. Further running `python avalanche_fit.py` will read the data in `avalanches/` and plot the avalanche size distributions. One can change the parameters and play with different settings of the neuristor network by modifying the file `main.py`. 

`python optimizer.py` uses reservoir computing to classify the MNIST handwritten digit dataset. Takes about 30 minutes in the test environment. 
Parameters of the reservoir (which is an array of thermal neuristors) are pre-tuned, while a fully-connected output layer is trained for 20 epochs. 

`python optimizer_KS.py` uses reservoir computing to predict the chaotic dynamics governed by the 2D Kuramoto-Sivashinsky equations. Takes about 3 minutes in the test environment. 
The trained reservoir weights, predictions and ground truths are saved in `ckpts/`, and one can further run `python visualize_trajectory.py` to plot the predicted vs. actual dynamics. 

The file `data.zip` contains raw experimental data in Fig. 1 and Fig. 5 from our paper. A description of the data is also provided within the zip file. 
