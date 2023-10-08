# Depression_Detection_from_Twitter_post

This repository is dedicated to the detection of depression using Twitter user's tweets. There are two types of tweets required for this project:

Random tweets that do not indicate depression.
  1. Tweets that suggest the user may have depression.
  2. The dataset for random tweets can be downloaded from Kaggle using this [link](https://www.kaggle.com/ywang311/twitter-sentiment/data).

Since there is no publicly available dataset specifically for depressive tweets, the essential dataset for this project was obtained using a web scraper called TWINT. This scraper collected tweets containing the keyword "depression" within a one-day span. It's worth noting that the scraped tweets may include both tweets indicating depression and unrelated tweets, such as those linking to articles about depression. To ensure better testing results, these scraped tweets were manually checked and the results were saved in CSV format as "depressive_tweets_processed.csv."

Additionally, the project utilizes pre-trained word vectors for the Word2Vec model provided by Google, which can be downloaded from this [link](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download).

The project employs a model architecture combining LSTM (Long Short-Term Memory) and CNN (Convolutional Neural Network). The model takes an input sentence, converts it into embeddings, and passes the resulting embedding vector through a convolutional layer. CNNs are effective at learning spatial structures from data, and this convolutional layer capitalizes on that capability to learn some structure from sequential data. Subsequently, the output of the convolutional layer is fed into a standard LSTM layer. Finally, the LSTM layer's output is processed by a standard Dense model for prediction.

The reported figures indicate that the model achieved an accuracy of 99% for tweets that do not indicate depression (labeled as zeros) and 97% for tweets that suggest depression (labeled as ones). These high accuracy scores suggest that the model is performing well in distinguishing between the two categories of tweets.

![Screenshot (48)](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/7004a6bb-cc7e-4f9b-996d-0059b6439db9)
![Screenshot (49)](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/90ff6970-abb6-415c-b6c6-9ca6452129be)
<div></div>

<div class="cell markdown" id="siqSAr_aOrf0">

## Output:

![model accuracy](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/32112381-6088-4d24-95ce-5deed0e117ad)
![model loss](https://github.com/Karunya003/Depression_Detection_from_Twitter_post/assets/85503646/ad0fa430-7755-4031-a7ab-afff5b66f300)
</div>
