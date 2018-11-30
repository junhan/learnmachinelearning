# AWS machine learning bootcamp notes

## vision video and image
serverless image processing workflow
https://github.com/aws-samples/aws-serverless-workshops/tree/master/ImageProcessing

utilize step functions and lamda

lamda execution time limit is now 15 minutes

video analysis
amazon rekognition video

## language and speech services
amazon transcribe
- mp3 input
- timestamping and confidence score
- speaker identification

amazon translate

amazon polly
- text to speech service

translate a chat channel using amazon 
twitch translate in real-time
and read out with the specified language using polly

amazon comprehend
- discover insights and relationships in text using deep learning
- topic modeling
- entities
- key phrases
- sentiment: negative, neutral, positive, mixed

amazon lex
- chatbot
- intent
- utterances

https://github.com/aws-samples/aws-ai-bootcamp-labs

## presentations
## Build a social media dashboard using machine learning and BI services
aws machine learning stack
AI services
- language: transcribe 

ML platforms: amazon sagemaker, aws deeplens, mechanical turks

ML frameworks: tensorflow, mxnet, pytorch, caffe2, chainer, horvod, gluon, keras

lamda function or serverless computing is another paradigm shift

aws glue: fully managed extract, transform, and load (ETL) service

word2vec: a group of related models that are used to produce word embeddings

word2vec deep learning

boto3: AWS SDK for Python, recommended python sdk for aws

[build a social media dashboard using machine learning and BI services](https://aws.amazon.com/blogs/machine-learning/build-a-social-media-dashboard-using-machine-learning-and-bi-services/)

aws cloudformation: work with stacks

## Build Computer Vision Applications with Amazon Rekognition
amazon rekognition image
amazon rekognition video

object detection
facial analysis
face comparison
facial recognition

interfacing with rekognition
s3 input for api calls, max image size 15mb

s3 event notification, lamda, step function

aws media analysis

aws connect: virtual call center and integrate some of the building blocks in speech service and existig machine learning services

## Build, Train, & Deploy ML Models Using SageMaker
watch youtube videos of Geoffrey Hinton - The Neural Network Revolution 

platform services: amazon sagemaker, emr, amazon mechanical turks

sagemaker
- hosted docker container
- hosted jupyter notebook
- building, training, hosting
- support distributed frameworks, e.g., tensorflow, mxnet, chainer, pytorch
- bring your own algorithm
- hyperparameter tuning: use naive bayesian network
- incremental training

(not tested yet) tpot for hyperparameter tuning

randall
randhunt@amazon.com

## Working with Amazon SageMaker Algorithms for Faster Model Training
built in sagemaker algorithms
- linear learner
- factorization machines
- k-means clustering
- principal component analysis (PCA)
- neural topic modeling
- deepAR: time series forecasting

## AWS DeepLens: A New Way to Learn Machine Learning

# AWS reinvent 2018 notes
several topics are interesting:
- serverless architecture based on lambda
- machine learning with sagemaker
- speech recognition and natural language processing

## serverless computing (lambda)
lambda is a shift of paradigm, and it supports auto-scaling and greatly decreases the development and deployment.

## machine learning using sagemaker
- machine learning training and deployment using sagemaker
- sagemaker supports the full lifecycle of machine learning workflow
- set up jupyter notebook and deploy the machine learning model to the sagemaker instances
- bring your own algorithm
- architecture recommendation (best practice): sagemaker instances exposes a model endpoint as the back-end, and then a lambda function is set up to invoke the model endpoint, and finally an api gateway is set up before the lambda function

## optimize machine learning flow
how to optimize the performance of overall machine learning procedure of model training and inferencing

model training
- model training is time-consuming, but once the training is done, the model can be used in various places to perform the inferencing and the inferencing takes 90% of the execution time.
- in the model training phase, choosing a proper model is the key. Although neural network can give the best accuracy, it is time consuming by nature. If random forrest/logistic regression or some ensembled form can provide slightly less accuracy, they can also be considered.
- hyperparameter tuning is important in the model training

model inferencing
- choosing a proper model can decrease the inferencing time greatly, e.g., decision tree is 1ms, random forrest is 10ms, while neural network is at 100ms level
- using GPU is preferred than CPU, but physical GPU is expensive
- `elastic inference` is introduced and can be attached to the docker instance managed by sagemaker, and this will future boost the inference performance while lowering the cost
- model inferencing can be further improved by using NVIDIA TensorRT. It takes deep learning model and combines the computation of certain layers and do not lose the precision. The performance gain is 70%.

Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow
- distributed deep learning
- 

## speech recognition and natural language processing (NLP) 
mxnet

### better analytics through natural language processing
analyze `all` data

structured data in relational database/nosql database and unstructured text in document and transcription
- learn machine learning
- specialize in nlp concepts
- prepare data
- train and re-train models
- deploy the model
- perform analysis
- state of the art algorithms and tools to conduct nlp

customized entities and classification
automatic training process
- train a classifier
- a text and a label
- automate algorithm selection
- automatic tuning and testing
- perform classification

customized classification
- automatic annotation of customized entities
- automate algorithm selection
- automatic tuning and testing

aws comprehend support custom classification and entities without nlp concepts

aws comprehend under the hood
mission is to remove ml/nlp complexity
- provide pre-trained model
- data collection and preparation
- algorithm selection
- testing and tuning
- deployment and management

FINRA financial regulatory agency
text mining
email, document, white paper, form, reference data
extract
commix command center for people and organization
identify the key information

amazon s3/input
amazon s3/matched
amazon elasticsearch service
amazon neptune: graph query, using rdf and support sparql

### sequence-to-sequence modeling with apache mxnet, sockeye and amazon sagemaker
sequence-to-sequence task can be used in the speech area

[aws labs for sockeye and mxnet](https://github.com/awslabs/sockeye)
understanding LSTM (long short-term memory)
- memory state being passed on over longer range
- provide solution against vanishing memory problem

encoder-decoder architecture for machine translation steps:
1. source language
2. encoder
3. though vector: capture the meaning instead of the word by word look up in dictionary
4. decoder
5. target language
it works well for short sentence, but it cannot handle long input.
Attention-based Encoder–Decoder Networks works better for longer sentence input.

`Seq2seq` model is the key model in natural language processing. Nlp area gains a lot of progress, when seq2seq with attention model is introduced and used in RNN. It takes source sequence and change to target sequence, and both sequences can be:
- word sequence to audio sequence (text to speech)
- audio sequence to word sequence (speech to text, speech recognition)
- word sequence in one language to another word sequence in another language (translation)

how attention works?
attention query based on past output and get attention score
RNN is used

in general, 80 - 200 words length is common for machine translation
below 80 works well
beyond 200 - 300 will use more tricks

package and algorithms
`mxnet sockeye package, sockeye is sequence-to-sequence machine translation toolkit`

nlp use cases:
- text classification
- document summarization
- language modeling
- caption generation
- speech recognition
- machine translation
- question answering

general nlp workflow
- tokenize
  tokenize source and target using `nltk/spaCy`
  and preprocess data using `sockeye`, the library used by amazon translate under the hood
- define model architecture (LSTM/GRU, layers, embedding size, hidden size, attention type)
- start training
- control model performance by measuring metrics on validation dataset
- perform inference
- sagemaker perform automatic model tuning for sequence-to-sequence, use bayesian optimization under the hood
- amazon sagemaker seq2seq (encoder-decoder with attention) algorithm is based on supervised training
- sagemaker can deploy the trained model and hosting, and it supports auto-scaling etc.

### resolving nlp problems using amazon sagemaker algorithms
- blazingText: can be used in Word2vec and text classification
  word2vec (unsupervised learning): generate high-quality distributed vectors, called word embedding. Thus capture the semantic relationships between words. This can be further used in the subsequent nlp processing pipeline.
  classification (supervised learning): transform to word vectors, along with label, and it can perform classification
- LDA: latent dirichlet allocation
- NTM: neural topic modeling, used in amazon comprehend
  topic modeling tries to find out the topics mentioned in the document, it is different from the simple classification, as the classification is only 
- sequence2sequence (seq2seq): amazon translate, transcribe, polly
  
nlp use cases and solution matrix

|   | NTM   | LDA   | Blaze  | Seq2Seq  |
|---|---|---|---|---|
| TTS (text to speech)  |   |   |   |  X |
| STT (speech to text)  |   |   |   |  X |
| ASR (automatic speech recognition)  |   |   |   | X  |
| CSR (continuous speech recognition)  |   |   |   | X  |
| OCR  |   |   |   | X  |
| Translation  |   |   |   | X  |
| Search  | X  | X  |   |   |
| Topic  |   |   |   | X  |
| Sentiment  | X  | X  |   |   |

[NLP workshop](https://github.com/aws-samples/aws-nlp-workshop)