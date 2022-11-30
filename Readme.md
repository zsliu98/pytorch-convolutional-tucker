# Introduction

Use Tucker-2 Decomposition on convolutional layer to compress CNNs.
We test this compression techniques on two data sets: FashionMNIST and CIFAR-10.

# Code Structure

- `trainer.py`: code for training and testing
- `tucker_layer.py`: implementation of compression of a convolutional layer
- `decomposition.py`: implementation of tucker decomposition algorithms: HOSVD and HOOI based on [tensorly](https://github.com/tensorly/tensorly)
- `vbmf.py`: Variational Bayesian matrix factorization, from [VBMF](https://github.com/CasvandenBogaard/VBMF)
- `AlexNet-MNIST.ipynb`: apply compression on AlexNet with FashionMNIST data set
- `AlexNet-CIFAR10.ipynb`: apply compression on AlexNet with CIFAR-10 data set

# Usage

Please follow the jupyter notebooks. Cause the code will read/write files to the disk and download/upgrade libraries, a cloud/virtual environment with a high-performance GPU is highly recommended.


# References

- Tamara G Kolda and Brett W Bader. 2009. Tensor decompositions and applications. SIAM review 51, 3 (2009), 455–500.
- Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, and Dongjun Shin. 2015. Compression of deep
convolutional neural networks for fast and low power mobile applications. arXiv preprint arXiv:1511.06530 (2015).