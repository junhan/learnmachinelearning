# machine learning research papers
## research paper: Distilling a Neural Network Into a Soft Decision Tree
combine a neural network into a decision tree, see [research paper by google brains](https://arxiv.org/abs/1711.09784)
 and [its pdf file download link](https://arxiv.org/pdf/1711.09784.pdf)

actual implementation is provided in [blog link](https://towardsdatascience.com/building-a-decision-tree-in-tensorflow-742438cb483e)

[github reference](https://github.com/benoitdescamps/Neural-Tree)

## A machine learning approach to prevent malicious calls over telephony networks
[link](https://arxiv.org/abs/1804.02566)
spammer, scammer or robodialer is a big issue in the telephony networks and it posed 

Prediction of a malicious call is a standard binary classification problem. 

call log data are collected from android app and two set of data are used in this experiment:
- historic information: basic info
- cross-referencing information: collect the stats within a time window (e.g., a month), convert sequence into a fixed size input vector

29 features are collected and normalized, and training set and test set are built upon them.

various machine learning models have been applied and evaluated:
- neural network: vanilla neural network from sklearn
- RNN: LSTM-base RNN using tensorflow
- support vector machine: SVM from sklearn
- random forrest: vanilla random forrest from sklearn and XGBoost
- logistic regression: vanilla from sklearn

Considering the data is highly skew, negative examples are 100x more than the positive ones. [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and AUROC (area under ROC) is used as an evaluation metrics, as AUROC is more robust to the data skew.

Incorporation of cross-referencing data greatly improve the prediction accuracy.

The best-performing model is a random forrest model trained by XGBoost, and it contains 100 decision trees with each having 3 layers. Within 29 features, only 21 features are chosen for prediction, and only top 10 are most used. Only using top 10 does not affect the prediction much and gained better performance.