The project is about performing sentiment analysis on tweets that are posted. We are handling a dataset of about 1.6 million tweets. The columns in the dataset include sentiment, ids
from which the tweet is being posted, date of the tweet, flag, user_name and the tweet. 

The dataset can be downloaded from kaggle( it is too big to be uploaded here). This is followed by loadig the dataset and naming the columns. Performing sentiment analysis required the
target sentiments and the tweet hence, the rest of the columns are dropped as they are not required. The data is grouped by 'sentiment' column, counting the number of occurences of each 
sentiment classifying 0s as negative and 4s as positive tweets. 

Text Preprocessing is traditionally an important step for Natural Language Processing (NLP) tasks. It transforms text into a more digestible form so that deep learning algorithms can 
perform better. Tweets usually contains a lot of information apart from the text, like mentions, hashtags, urls, emojis or symbols. Since normally, NLP models cannot parse those data,
we need to clean up the tweet and replace tokens that actually contains meaningful information for the model.
The Preprocessing steps taken are:
Lower Casing: Each text is converted to lowercase.
Replacing URLs: Links starting with 'http' or 'https' or 'www' are replaced by '<url>'.
Replacing Usernames: Replace @Usernames with word '<user>'. [eg: '@Kaggle' to '<user>'].
Replacing Consecutive letters: 3 or more consecutive letters are replaced by 2 letters. [eg: 'Heyyyy' to 'Heyy']
Replacing Emojis: Replace emojis by using a regex expression. [eg: ':)' to '<smile>']
Replacing Contractions: Replacing contractions with their meanings. [eg: "can't" to 'can not']
Removing Non-Alphabets: Replacing characters except Digits, Alphabets and pre-defined Symbols with a space.
As much as the preprocessing steps are important, the actual sequence is also important while cleaning up the text. For example, removing the punctuations before replacing the urls means
the regex expression cannot find the urls. Same with mentions or hashtags. So make sure, the actual sequence of cleaning makes sense.

After the preprocessing done, the data is all cleaned up and in the format which can be used. This is followed by creation of word cloud of both positive and negative words. This is 
followed by splitting of dataset into training and testing. This code sets up a machine learning pipeline using CountVectorizer and MultinomialNB to classify tweet sentiments. 
It employs GridSearchCV to perform hyperparameter tuning, testing different configurations of binary count vectors, n-gram ranges, and smoothing parameters to find the optimal model. 
The process starts by recording the time, defining the pipeline, setting the parameter grid, and running the grid search with cross-validation. Finally, it fits the model to the
training data and prints the training duration, best cross-validation score, and the best parameters. This ensures the model is fine-tuned for the best performance on the sentiment 
classification task.
Naive Bayes model for sentiment analysis proves to be a good one with an accuracy of about 0.8. In case of this model, it considers each word in the tweet to be independent of itself 
and predict the sentiment of the tweet. We have performed descriptive analysis on the model to check for parameters such precision, accuracy and recall and also plotted confusion matrix.
After that we gave our own inputs as tweets and asked the model to predict if it is positive or negative.
