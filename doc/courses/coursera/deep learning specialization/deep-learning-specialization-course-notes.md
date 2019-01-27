# Neural Networks and Deep Learning
## Introduction to deep learning 
artificial neural network (ANN)

| type  | description  |
|---|---|
| standard NN | input, one hidden layer, and one output |
| convolutional NN | the key word is the convolution, i.e., multiple inputs are convoluted into one result  |
| recurrent NN | map an input sequence to an output sequence, audio to text, text to audio, words to words |

although a shallow neural network (with 1 hidden layer) can approximate any function, i.e. can in principle learn anything. But the deep neural network tends to works better in real world problems.

structured data (stat, gpd data, price) vs unstructured data (image, audio, text)

supervised learning vs unsupervised learning

## Neural Networks Basics
goals
- logistic regression model, structured as a shallow neural network

### Logistic Regression as a Neural Network
logistic regression is for binary classification

classify whether an image is a cat (1) or not (0)
an input image is represented as 64x64 pixels of 3 basic colors (Red, Green, Blue), so $x$ vector has $64 * 64 * 3 = 12288$ dimensions (denoted as $n_x$), and this classification problem can be represented as $X -> Y$:
- $m$ is the size of training or testing set
- label $y = \{1, 0\}$, $Y = [y_1, y_2, ..., y_m]$ and $Y$ is $1 * m$ dimension matrix
- $X = [x_1, x_2, ..., x_m]$, $X$ is $n_x * m$ dimension matrix and $x$ is feature vector.

stack training and testing data in **columns** when constructing matrix.

logistic regression
given $x$, find out $\hat{y}=P(y = 1 | x)$, the probability of $y = 1$ given input $x$.

$y = w^Tx + b$ can be any value, and $0 <= \hat{y} <= 1$, so use a sigmoid function to limit the result between 0 and 1.

logistic regression will find out weight $w$ and $b$ based on the training data.

sigmoid function $\sigma(z) = 1 / (1 + e^{-z})$
- when $z \to \infty$, $\sigma(z) = 1$
- when $z \to -\infty$, $\sigma(z) = 0$

The **loss/error function** computes the error for a single training example; the **cost function** is the average of the loss functions of the entire training set.
- loss function $L(\hat{y}, y) = -(y\log(y) + (1-y)\log(1-y))$
- cost function $J(w, b) = 1/m \Sigma{L(\hat{y}, y)}$

gradient descent

tips for highly efficient calculation
- remove unnecessary for loops
- vectorize code by stacking (1) put all the input features in rows and (2) stacking training example in columns, so that matrix operation can handle them in one run
- use python's `broadcasting` feature with care, and use `reshape()` function and `assert` if necessary.
```
>>> d
array([[2.25300834, 1.67936284, 3.01856513],
       [2.79112261, 2.15177373, 1.90725032],
       [5.00728627, 5.98184963, 0.25372975]])
>>> f = d + 1
>>> f
array([[3.25300834, 2.67936284, 4.01856513],
       [3.79112261, 3.15177373, 2.90725032],
       [6.00728627, 6.98184963, 1.25372975]])
```

numpy matrix dot and multiply differences
- dot: matrix product
```
np.dot is the dot product of two matrices.
|A B| dot |E F| = |A*E+B*G A*F+B*H|
|C D|     |G H|   |C*E+D*G C*F+D*H|
```
- multiply (*): element-wise multiplication
```
Whereas np.multiply and * operation does an element-wise multiplication of two matrices.
|A B| multiply |E F| = |A*E B*F|
|C D|          |G H|   |C*G D*H|
```

sigmoid function (or logistic function) is widely used in machine learning and deep learning. Use np.exp instead of python's built-in math.exp(x), as np.exp can take in a vector or matrix while math.exp can only handle real numbers.
$sigmoid(x) = \frac{1}{1 + e^{-x}}$

sigmoid gradient is calculated as follows:

$sigmoid(x) = s = \frac{1}{1 + e^{-x}}$

let $t = e^{-x}$, so sigmoid function is now $s = \frac{1}{1+t}$, therefore, $\frac{ds}{dx}$ can be calculated as

$\frac{ds}{dx} = \frac{ds}{dt}  \times \frac{dt}{dx}$

based on calculus chain rule and division rule $(\frac{f}{g})'=\frac{f'g - fg'}{g^2}$

$$
\frac{ds}{dx} = \frac{ds}{dt}  \times \frac{dt}{dx} \newline
= (\frac{1}{1+t})' \times \frac{dt}{dx} \newline
= -\frac{1}{(1+t)^2} \times \frac{dt}{dx} \newline
= -\frac{1}{(1+e^{-x})^2} \times -e^{-x} \newline
= \frac{e^{-x}}{(1+e^{-x})^2} \newline
= \frac{1 + e^{-x} - 1}{(1+e^{-x})^2} \newline
= \frac{1}{1+e^{-x}} - (\frac{1}{1+e^{-x}})^2\newline
= s \times (1-s)
$$

norm or distance
- L1-norm, Manhattan distance, $||x||_1=\Sigma{|x|}$
- L2-norm, Euclidean norm, Euclidean distance, $||x||_2=\sqrt{\Sigma{x^2}}$

L1-norm, L2-norm as as loss function
- L1-norm loss function, least absolute deviations (LAD), least absolute errors (LAE)
- L2-norm loss function, least square error

# Improving Deep Neural Networks Hyperparameter tuning Regularization and Optimization
## bias and variance
- bias: The bias is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (under-fitting).
- variance: The variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (over-fitting).
- high bias
  - under-fitting, cannot fit well the training set.
  - try to solve the high bias first with an acceptable error rate
  - base error (bayes error) rate
  - how to solve it?
    - use bigger network
    - more complex neural network structures
    - train longer
- high variance
  - over-fitting, cannot generalize over training set and will not do well in the dev set or test set
  - how to solve high variance?
    - get more data
    - regularization
    - more complex neural network structures
- bias variance trade-off
  - these two goals are conflicted with each other often
  - but for neural network, it can achieve both low bias and low variance
  - the reason that neural network gains popularity

## regularization
regularization may increase bias a little, but can reduce high variance or over-fitting problem. It prevents the coefficients to fit so perfectly that the model leads to over-fitting.

- L2 regularization: sum of square of the weights.
$\frac{\Lambda}{2m}||W||^2_2=\frac{\Lambda}{2m}\Sigma{W_i^2}=\frac{\Lambda}{2m}W^TW$
- L1 regularization: sum of weights
$\frac{\Lambda}{2m}\Sigma{W_i} = \frac{\Lambda}{2m}||W||_1$

$\lambda$ is the regularization parameter, i.e., another hyperparameter to consider for optimization purpose

## dropout regularization
dropout is another popular form of regularization and it is mainly used in neural network.

use it in the training set, ignore the dropout in the dev/test set

inverted dropout implementation

the intuition behind why dropout works is because by dropping out certain neurons randomly reduces the dependency to certain features. For example, in a face recognition problem, the classifier should not rely too much on certain features, e.g., if mouse is covered, the overall classifier (eye, ears, head) should work.

## other regularization technique
data augmentation, generating more fake training set by tweaking the input. e.g., flipping upside down, distortion

early stopping of gradient descent optimization

## set up optimization problem
normalization and standardization

normalize training set
- the mean $\mu = \frac{1}{m}\Sigma{x}$
- the normal variance standard deviation $\sigma=\frac{1}{m}\Sigma{(x - \mu)}^2$

vanishing/exploding gradients 

numerical approximation of gradients

$df/d\theta = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$

## optimization algorithms
- mini-batch gradient descent
  - 1 epoch: 1 single pass of 1 step of mini-batch gradient descent through training set
  - if mini-batch size = m (the whole training set size), it is batch gradient descent
  - if mini-batch size = 1, it is stochastic gradient descent
- stochastic gradient descent
  - use 1 training sample at one iteration
- bias correction
- momentum
  - gradient descent with momentum
  - Momentum takes into account the past gradients to smooth out the update and reduce the oscillations
  - compute $dw$, $db$ on mini-batch, then $v_{dw}=\beta * v_{dw} + (1 - \beta) * v_{dw}$
  - smooth out the gradient
  - instead of following gradient direction, the update on $w$ and $b$ are calculated based on momentum
- RMSProp: root mean square property
- Adam optimization: combine
  - Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp (described in lecture) and Momentum.
  - It calculates an exponentially weighted average of past gradients, and stores it in variables  v  (before bias correction) and  $v^{corrected}$ c(with bias correction).
  - It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables  s  (before bias correction) and  $s^{corrected}$  (with bias correction).
  - It updates parameters in a direction based on combining information from "1" and "2".

## hyperparameter algorithms
- batch normalization
  - normalization can speed up learning
  - normalize $z^{(i)}$
- batch normalization
  - make sure that the input to the neural network will not change too much
- multi-class classification
  - not binary classification
  - softmax layer
  - softmax activation function, take n input and produce a vector of n output
  - loss function

# Structuring Machine Learning Projects
## ideas of machine learning strategy
ideas:
- collect more data
- collect more diverse training set
- train algorithm longer with gradient descent
- try Adam instead of gradient descent
- try bigger network
- try smaller network
- try dropout
- add $L_2$ regularization
- change network architecture
  - activation function
  - ...

evaluation matrix
- precision:  positive predictive value, fraction of relevant instances among the retrieved instances
- recall: sensitivity, the fraction of relevant instances that have been retrieved over the total amount of relevant instances
- F-1 score: a combination of precision and recall

Satisficing and optimizing metric

why compare to human-level performance:
- get labeled data from humans
- gain insight from manual error analysis
- better analysis of bias/variance

problems that ML does better than human:
- online advertising
- product recommendation
- logistics (predicting transit time)
- loan approvals

These are structured data and not natural perception (vision, hearing):
- speech recognition
- image recognition
- medical

two fundamental assumptions of supervised learning:
- fit training set pretty well (bias)
- the training set performance generalizes pretty well to teh dev/test set (variance)

reducing (avoidable) bias and variance
- human-level
- training error
- dev error

reduce avoidable bias
- train bigger model
- train longer/better optimization algorithms
  - momentum, RMSprop, Adam
- NN architecture/hyperparameter search
  - RNN, CNN

reduce variance:
- more data
- regularization
  - l2, dropout, data augmentation
- NN architecture/hyperparameter search

Build the first system quickly and then iterate:
- set up dev/test set and metric
- build initial system quickly
- use bias/variance analysis and error analysis to prioritize next steps

bias/variance on mismatched training and dev/test sets:
- avoidable bias between human level and training set error.
- variance between training set and training-dev set
- data mismatch between training-dev set and dev set
- degree of over-fitting to the dev set between dev and test set

Transfer Learning vs Multitask Learning：
- Multitask Learning: for several similar tasks, optimizing one task may help another, so optimize them at the same time
- transfer learning: transfer one pre-trained network to another task

# Convolutional Neural Networks
## edge detection
- vertical edge detection
  - $$
\left(\begin{array}{cc} 
1 & 0 & -1\\ 
1 & 0 & -1\\
1 & 0 & -1\\
\end{array}\right)
$$ 
  - python convolution operator
  - tensorflow conv2d
- horizontal edge detection
  - $$
\left(\begin{array}{cc} 
1 & 1 & 1\\ 
0 & 0 & 0\\
-1 & -1 & -1\\
\end{array}\right)
$$

the basic convolution operation (*) is the key for backward propagation to learn
- it shrinks the image output
- throw away a lot of information from the edges/corners

to address these two issues, pad the image with one layer of 0, e.g., the original image is 6 x 6, pad the image to become 8 x 8, so that the convoluted image preserves the original size 6 x 6. the padding (p) is one hyperparameter of convolutional network.

valid and same convolutions, original image and filter (f x f)
- **valid**: no padding is used, perform the normal convolution operation
  $(n \times n) * (f \times f) -> (n - f + 1 * n - f + 1)$
- **same**: pad (p) so that output image size is the same as the input size
  $n + 2 \times p - f + 1 = n  \implies p = \frac{f-1}{2}$

strided convolutions operation
stride = 2, move left and move down the original image with the size of provided stride size instead of default 1

basic convolutions elements with strided convolutions:
- (n x n) image
- (f x f) filter
- padding p
- stride s
- output image size: using floor operation $\lfloor \frac{n + 2p - f}{s} + 1 \rfloor \times \lfloor \frac{n + 2p - f}{s} + 1 \rfloor$

convolutions on RGB image
- use a filter 3x3x3 (height x width x channel) to perform convolution operation against the original 6x6x3 image and get 4x4 result
- multiple filters will produce 4x4x$n_c^\prime$, i.e., the outputs can be stacked, where $n_c^\prime$ is the number of filters.
- source $n \times n \times n_c * f \times f \times n_c \implies n - f + 1 \times n - f + 1 \times n_c^\prime$, where $n_c^\prime$ is the number of filters, assume s = 1 and p = 0

in one convolution layer, 10 filters with 5x5 RGB image, the total parameter is $(5 \times 5 \times 3 + 1) * 10 = 7600$, where parameter is weight in the filter plus one bias parameter. The parameter size in the convolution layer will not be affected by the input image.

notation for layer l as a convolution layer
- $f^{[l]}$ = filter size or dimension, e.g., 3x3
- $p^{[l]}$ = the amount of padding (valid padding p = 0 or same)
- $s^{[l]}$ = stride
- $n_c^{[l]}$ = number of filters, also number of **output** channels
- each filter is $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$
- activations: $a^{l} -> n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$
  - with vectorization: $A^{l} = m \times n_H^{[l]} \times n_W^{l]}[ \times n_c^{[l]}$
- Weights: weight parameters in all the filter, $f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$
- bias: $n_c^{[l]} - (1,1,1,n_c^{[l]} )$
- input: $n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}$
- output: $n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$

typical types of layer in a convolutional network:
- convolution layer (conv)
- pooling layer (pool)
- fully connected layer (fc)
  
pooling layer: max pooling and average pooling
- extract the max of a 2x2 region from a filter
- it keeps the same amount of channels
- stride controls the steps to jump

hyperparameters in pooling layer:
- f: filter size
- s: stride
- max or average pooling
- the rule to calculate the dimension in pooling layer is the same as the convolution layer
- no parameters to learn! nothing for gradient descent to learn
- pooling is done independently in each channel, so the number of output channel is the same as input

# reference
this git repo contains the questions and answers listed in the specialization courses. [see reference](https://github.com/Kulbear/deep-learning-coursera)
