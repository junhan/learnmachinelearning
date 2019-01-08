# Introduction to deep learning 
artificial neural network (ANN)

| type  | description  |
|---|---|
| standard NN | input, one hidden layer, and one output |
| convolutional NN | the key word is the convolution, i.e., multiple inputs are convoluted into one result  |
| recurrent NN | map an input sequence to an output sequence, audio to text, text to audio, words to words |

although a shallow neural network (with 1 hidden layer) can approximate any function, i.e. can in principle learn anything. But the deep neural network tends to works better in real world problems.

structured data (stat, gpd data, price) vs unstructured data (image, audio, text)

supervised learning vs unsupervised learning

# Neural Networks Basics
goals
- logistic regression model, structured as a shallow neural network

## Logistic Regression as a Neural Network
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
- L1-norm, Manhattan distance $||x||_1=\Sigma{|x|}$
- L2-norm, Euclidean norm, Euclidean distance $||x||_2=\sqrt{\Sigma{x^2}}$

# reference
this git repo contains the questions and answers listed in the specialization courses. [see reference](https://github.com/Kulbear/deep-learning-coursera)
