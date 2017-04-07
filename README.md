# ECE 408 Project
The goal of this project is to accelerate the forward propagation step of the Convolutional Neural Network (CNN) algorithm using GPU. The sequential implementation provided follows the basic algorithm 16.4 and 16.5 decribed in [book chapter 16](https://wiki.illinois.edu/wiki/display/ece408f16/Book+Chapters?preview=/602518692/603851747/3rd-Edition-Chapter16-case-study-DNN-FINAL.pdf). The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

Our team ranked 2nd in final project competition of 16fall ECE 408. The sequential implementation takes about 30 minutes for the largest data set (with 10,000 images), while after GPU-accelerated, it takes around 200ms only.

This project previously ran on RAI system provided by instructors. Thus this project may be more worthwhile as a code reference.

### Major Optimization
（Welcome to check the <a href="./report.pdf">report</a> for detailed analysis :D）
1. Convolution with unrolled matrix multiplication. The conventional convolution calculation is not suitable for GPU code in consideration of [memory coalescing](https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/). In contrast, unrolled matrix multiplication, which involved little extra work though, can achieve better memory coalescing as neighboring thread can read continus stride of data.
2. Tiled implementation. This is a commonly used trick in [matrix multiplication](http://www.techdarting.com/2014/03/matrix-multiplication-in-cuda-using.html). Threads in a same tile can load data to shared memory together, which highly reduce the global memory bandwidth.
3. Dimension transformation. The original data layout provided is not suitable for memory coalescing, thus we transorm some dimension in input and output of intermediate functions.

### Team Members
[Xiaocong Chen](https://www.linkedin.com/in/xiaocongchen/)
<br>
[Xinzhou Zhao](https://www.linkedin.com/in/xinzhou-zhao-9a2406103/)
<br>
Tianyi Shan
