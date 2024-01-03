# Collective dynamics and long-range order in thermal neuristor networks
This repository contains source code for reproducing the results in the paper *Collective dynamics and long-range order in thermal neuristor networks*. ArXiv link: [arXiv:2312.12899](https://arxiv.org/abs/2312.12899v1). 

Here is another relevant paper on thermal neuristors: [Reconfigurable Cascaded Thermal Neuristors for Neuromorphic Computing](https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.202306818)

Requirements: NumPy, PyTorch >= 2.0

Usage: 

`python main.py` computes the avalanche size distribution with specific parameters. One can change the parameters by modifying the file `main.py`. 

`python optimizer.py` uses reservoir computing to classify the MNIST handwritten digit dataset. Parameters of the reservoir (which is an array of thermal neuristors) are pre-tuned, while a fully-connected output layer is trained for 20 epochs. 
