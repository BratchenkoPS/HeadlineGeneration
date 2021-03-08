# HeadlineGeneration
This repo a solution to a test task where you have to show ability to solve headline generation
 problem using [RIA_news dataset](https://github.com/RossiyaSegodnya/ria_news_dataset). <br />
## 1) Model choice and general approach
So first of all let's start with the choice of algorithm for the task. There are several most common 
approaches: <br />
1) Seq2seq transformer architecture
2) Seq2seq RNN's
3) LSA/HDP 


We probably not gonna use transformers since we are fairly limited in terms of computing power.
LSA/HDP are also not very robust from my experience and more suited for summarizing task.
 So the only option left is RNN's. But we still got 
over 1 million replics to work with and RNN's a known to be fairly slow due to their sequential nature.
What could we do? Search for faster models!
After some research i found [this paper](https://arxiv.org/abs/1705.03122) about convolutional seq2seq
architecture which promises to be 10x faster than LSTM while beating it in most tasks. Sounds good! <br />

But we still got computing power issue: there is a large open domain dataset which would require a huge model to learn. 
And although convolutional seq2seq model is way faster than the LSTM it would take more than a couple of days to get
the parameters right and train it. And we don't really have that much time. <br /> 
The solution that i came it with is fairly simple: just pick a certain domain and train a model for that domain!
How we gonna do that? By using unsupervised clustering via Gaussian Mixtures. We also gonna limit the text length to 100
 tokens to make the model lighter.
 
 The pipeline is gonna look like that:
 1) Load the file, pick a certain amount of examples with length restriction for clusterization (100k should be enough, otherwise it would take too long to compute)
 2) Clean it with regex from parsing artifacts and remove punctuation
 3) Build embeddings using [Universal Sentence Encode](https://tfhub.dev/google/universal-sentence-encoder/4)
 4) Build TSNE embeddings
 5) Build clusters and visualize it
 6) Pick the biggest cluster
 7) Train the model on the data from biggest cluster
## 2) Results
So the in the result we are getting around 10k replics that supposedly belong to the same domain area and train 
the model on it. The best result metric (=*mean(Rouge-1-F, Rouge-2-F, Rouge-L-F)*) that i managed to get on such a small dataset is is 0.1617.
The is also seems to be a little bit of a leak between train and test data, where we have examples with same titles
but slightly different texts and vice versa. That might also inflate a metric a little bit but that's just the nature of open-domain datasets. Still not bad for such a small amount of data!
## 3) Improvements.
There are a lot of ways to improve the model that i've just created if we still want to train a model for a
certain domain area even if not going transformer's way (which would probably be the best way anyway). Here are some of them:
1) Apparently clustering on title does not work very well on titles. The biggest cluster seems to be a mix of everything in the middle 
(you can take a look at clustering picture at **result/clusters.html** after running **main.py**, note that there is only 1/10 of data
ploted so it would be easier to analyse). We should try clusterization on text instead of titles and pick clusters
manually analyzing the result data. We can also use keyword search to pick domain area like "sport", "politics", etc.
2) Use more data. This seems to be the biggest issue, because the model seems to be overfitting pretty fast. 
If we will use more data we could also leave the punctuation, use longer sequences and stack a bit more layers.
Should also probably increase the kernel size for that one to find patterns in longer sequences.
3) Use pretrained fasttext/word2vec embeddings vocab as a baseline for embedding layers.
4) All of TODO's from scripts and more docstring/typization hints
## 4) How to run
```git clone https://github.com/BratchenkoPS/HeadlineGeneration``` <br />
```cd HeadlineGeneration``` <br />
```python -m venv /venv``` <br />
```pip install -r requirements.txt``` <br />
```python main.py```


 
