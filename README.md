# Depression_Detection_from_Twitter_post

This repository is dedicated to the detection of depression using Twitter user's tweets. There are two types of tweets required for this project:

Random tweets that do not indicate depression.
  1. Tweets that suggest the user may have depression.
  2. The dataset for random tweets can be downloaded from Kaggle using this [link](https://www.kaggle.com/ywang311/twitter-sentiment/data).

Since there is no publicly available dataset specifically for depressive tweets, the essential dataset for this project was obtained using a web scraper called TWINT. This scraper collected tweets containing the keyword "depression" within a one-day span. It's worth noting that the scraped tweets may include both tweets indicating depression and unrelated tweets, such as those linking to articles about depression. To ensure better testing results, these scraped tweets were manually checked and the results were saved in CSV format as "depressive_tweets_processed.csv."

Additionally, the project utilizes pre-trained word vectors for the Word2Vec model provided by Google, which can be downloaded from this [link](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download).

The project employs a model architecture combining LSTM (Long Short-Term Memory) and CNN (Convolutional Neural Network). The model takes an input sentence, converts it into embeddings, and passes the resulting embedding vector through a convolutional layer. CNNs are effective at learning spatial structures from data, and this convolutional layer capitalizes on that capability to learn some structure from sequential data. Subsequently, the output of the convolutional layer is fed into a standard LSTM layer. Finally, the LSTM layer's output is processed by a standard Dense model for prediction.

<div class="cell markdown" id="lib">

### Import Necessary Library

</div>
<div class="cell code" data-execution_count="1" id="import">
 In this section, necessary Python libraries and modules are imported. These libraries include:

* **warnings:** Used to manage and suppress warning messages.
* **ftfy:** A library for fixing Unicode text.
* **matplotlib.pyplot:** Used for creating plots and graphs.
* **nltk:** The Natural Language Toolkit for text processing.
* **numpy:** A library for numerical computations.
* **pandas:** Used for data manipulation and analysis.
* **re:** Regular expression module for text processing.
* Various modules from **Keras** for building deep learning models.
* Modules from **scikit-learn (sklearn)** for machine learning.
* **KeyedVectors** from Gensim for working with pre-trained word embeddings.

``` python
import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re

from math import exp
from numpy import sign

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model

# Reproducibility
np.random.seed(1234)

```
</div>

<div class="cell markdown" id="constant">

### Constants and Hyperparameters

</div>
<div class="cell code" data-execution_count="2" id="con">
This section defines various constants and hyperparameters used throughout the code. These include the number of rows to read from datasets, the maximum sequence length for tweets, the maximum number of words in the tokenizer, embedding dimensions, train-test split ratios, learning rate, and the number of training epochs.

``` python
DEPRES_NROWS = 3200  # Number of rows to read from DEPRESSIVE_TWEETS_CSV
RANDOM_NROWS = 12000 # Number of rows to read from RANDOM_TWEETS_CSV
MAX_SEQUENCE_LENGTH = 140 # Max tweet size
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS = 10
```
</div>

<div class="cell markdown" id="loading">

### Data Loading and Word Embeddings: 

</div>
<div class="cell code" data-execution_count="3" id="embedding">
  
**Data Loading:**
  The code loads two datasets: 
  * depressive_tweets_processed.csv for depressive tweets 
  * Sentiment Analysis Dataset 2.csv for random tweets. 
  These datasets are read into Pandas DataFrames.

**Word Embeddings:** Word embeddings are loaded from the pre-trained Word2Vec model stored in the EMBEDDING_FILE. These embeddings are used to initialize the embedding layer of the neural network model.

``` python
# Data Loading
df = 'depressive_tweets_processed.csv'
RANDOM_TWEETS_CSV = 'Sentiment Analysis Dataset 2.csv'
depressive_tweets_df = pd.read_csv(df, sep='|', header=None, usecols=range(0,9), nrows=DEPRES_NROWS)
random_tweets_df = pd.read_csv(RANDOM_TWEETS_CSV, encoding="ISO-8859-1", usecols=range(0,4), nrows=RANDOM_NROWS)
# Word Embeddings
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
```
</div>

<div class="cell markdown" id="loading">

### Data Loading and Word Embeddings: 

</div>
<div class="cell code" data-execution_count="4" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="embedding" data-outputId="f4d92e86-eab0-4066-e157-4ec730618d5d">
This section focuses on preparing the text data for modeling:

Text preprocessing functions are defined to clean and preprocess tweet text.
Tweets are cleaned to remove URLs, hashtags, mentions, emojis, and punctuation. Contractions are expanded, stop words are removed, and words are stemmed.
Tokenization and sequence padding are performed on the cleaned tweets using Keras' Tokenizer and pad_sequences functions.

``` python
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            # remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())

            # fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)

            # expand contraction
            tweet = expandContractions(tweet)

            # remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(tweet)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            # stemming words
            tweet = PorterStemmer().stem(tweet)

            cleaned_tweets.append(tweet)

    return cleaned_tweets
depressive_tweets_arr = [x for x in depressive_tweets_df[5]]
random_tweets_arr = [x for x in random_tweets_df['SentimentText']]
X_d = clean_tweets(depressive_tweets_arr)
X_r = clean_tweets(random_tweets_arr)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_d + X_r)

sequences_d = tokenizer.texts_to_sequences(X_d)
sequences_r = tokenizer.texts_to_sequences(X_r)


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)
data_r = pad_sequences(sequences_r, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data_d tensor:', data_d.shape)
print('Shape of data_r tensor:', data_r.shape)
``` 
</div>

<div class="cell markdown" id="model">
  
### Model Construction 

</div>
<div class="cell code" data-execution_count="5" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="model" data-outputId="model">
In this section, the deep learning model is constructed using the Keras Sequential API. Here's a breakdown of the model architecture:

**Embedding Layer:**

This layer initializes word embeddings using pre-trained word vectors (embedding_matrix) obtained earlier. It ensures that the word embeddings are fixed during training (set trainable=False).
input_length is set to MAX_SEQUENCE_LENGTH, which is the maximum sequence length of the padded tweets.

**Convolutional Layer:**

A 1D convolutional layer is added with 32 filters and a kernel size of 3.
The activation function is ReLU (Rectified Linear Unit).
padding='same' ensures that the input and output dimensions match.
Max-pooling is applied with a pool size of 2.
Dropout with a rate of 0.2 is used for regularization.

**LSTM Layer:**

A Long Short-Term Memory (LSTM) layer with 300 units is added.
Another dropout layer with a rate of 0.2 is used for regularization.

**Dense Layer:**

A dense layer with a single neuron and a sigmoid activation function is added for binary classification.
The model is compiled with binary cross-entropy loss and the Nadam optimizer.

``` python
# Model Architecture
model = Sequential()
# Embedded layer
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False))
# Convolutional Layer
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
# LSTM Layer
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
print(model.summary())
```
</div>

<div class="cell markdown" id="train">
  
### Model Training

</div>
<div class="cell code" data-execution_count="6" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="train" data-outputId="train">
In the training section:

* EarlyStopping is used as a callback to monitor the validation loss (val_loss) and stop training if it does not improve for a specified number of epochs (patience=3).
* The model.fit() function is called to train the model on the training data (data_train and labels_train) and validate it on the validation data (data_val and labels_val).
* Training is performed for the specified number of epochs (EPOCHS), with a batch size of 40 and shuffling the data during training.

``` python
early_stop = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val), epochs=EPOCHS, batch_size=40, shuffle=True, callbacks=[early_stop])
```
</div>

<div class="cell markdown" id="evaluation">
  
### Plotting and Model Evaluation

</div>
<div class="cell code" data-execution_count="7" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="evaluation" data-outputId="evaluation">
In this section, the code evaluates the trained model and plots the training and validation accuracy and loss curves:

* The training and validation accuracy and loss curves are plotted using Matplotlib.
* The model is used to predict labels on the test data (data_test), and the predicted labels are rounded to obtain binary predictions.
* The accuracy of the model is calculated using scikit-learn's accuracy_score function and printed.
* A classification report is generated and printed, providing additional metrics such as precision, recall, and F1-score for binary classification.
* These steps allow you to assess the performance of the deep learning model for depression detection based on the provided datasets and settings.

``` python
# Plotting
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Model Evaluation
labels_pred = model.predict(data_test)
labels_pred = np.round(labels_pred.flatten())
accuracy = accuracy_score(labels_test, labels_pred)
print("Accuracy: %.2f%%" % (accuracy*100))

print(classification_report(labels_test, labels_pred))

```
</div>


The reported figures indicate that the model achieved an accuracy of 99% for tweets that do not indicate depression (labeled as zeros) and 97% for tweets that suggest depression (labeled as ones). These high accuracy scores suggest that the model is performing well in distinguishing between the two categories of tweets.

![Screenshot (48)](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/7004a6bb-cc7e-4f9b-996d-0059b6439db9)
![Screenshot (49)](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/90ff6970-abb6-415c-b6c6-9ca6452129be)
<div></div>

<div class="cell markdown" id="output">

## Output:

![model accuracy](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/32112381-6088-4d24-95ce-5deed0e117ad)
![model loss](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/ad0fa430-7755-4031-a7ab-afff5b66f300)
</div>
