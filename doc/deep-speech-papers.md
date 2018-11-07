# deep speech by baidu research and Andrew Ng
[baidu research homepage for deep speech](http://research.baidu.com/Blog/index-view?id=90)

Baidu research developed an end-to-end automatic speech recognition utilizing deep learning approaches.

## deep speech v1 2014
[Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567).
deep learning approach is used in automatic speech recognition area, to be more specific, a well-optimized RNN runs on multiple GPUs and greatly simplifies the whole pipeline of ASR.

## deep speech v2 2015
[Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
a CTC is used

## deep speech v3 2017
[Exploring Neural Transducers for End-to-End Speech Recognition](https://arxiv.org/abs/1707.07413)

# mozilla deep speech implementation
[implementation of deep speech from mozilla machine learning team](https://github.com/mozilla/DeepSpeech)
mozilla machine learning group implements deep speech v2 paper using tensorflow and reproduce the result of baidu's research paper

Side note, this situation like Apache's implementation of Google's 3 papers on big data.
| Google file system (2003)  | Apache Hadoop file system (HDFS) |
| MapReduce (2004)           | Apache Hadoop                    |
| BigTable (2006)            | Apache HBase                     |

## A Journey to <10% Word Error Rate (December 2017)
[A Journey to <10% Word Error Rate](https://hacks.mozilla.org/2017/11/a-journey-to-10-word-error-rate/)
A simple spell checker is first used, however, it does not provide