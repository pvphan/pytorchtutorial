# PyTorch Tutorial

![mnist_image](https://abpaudel.com/assets/img/posts/mnist.png)
(Figure from Abhishek Paudel [5], linked below)

This template provides a quick way to experiment with neural network architectures on the classic 'MNIST database of handwritten digits' using PyTorch. Library dependencies and dataset parsing are handled for you.

Prerequisites on your local machine are `docker` and `make`.


### Quick start:

1. Clone this repo and `cd` into it.
2. Modify (only) the class `MnistModel` in `mnist.py`. You can use `pytorch.py` as an example.
3. In console, run `make runmnist` to train and test the network
4. Check your results at the end, it will print `Correctly predicted __%` to tell you how the network performed!


### References:

1. (dataset) [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
2. (videos) [3Blue1Brown: Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1&ab_channel=3Blue1Brown)
3. (tutorial) [Linear Regression with PyTorch](https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817)
4. (tutorial) [PyTorch basics](https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29)
5. (blog) [Abhishek Paudel: Pen Stroke Sequence Feature Extraction from MNIST Digits](https://abpaudel.com/blog/mnist-sequence-feature-extraction/)
