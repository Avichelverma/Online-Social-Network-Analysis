"""
classify.py
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
import re
from nltk.tokenize import word_tokenize


def load_data():
    """
    Load the tweets pickle data collected from collect.py
    """

    pkl_file = open('tweets.pkl', 'rb')
    tweets = pickle.load(pkl_file)
    pkl_file.close()

    return tweets


def clean_data(tweets):
    """
    Params: Tweets
    Returns: Clean text.... Preprocessed file after cleaning the stopwords, punctuations, html, https.
    """

    clean_text = []

    for tweet in tweets:
        clean_tweet = ' '.join(re.sub(r'(?:\@|https?\://)\S+|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',
                                              " ", tweet.lower()).split())
        clean_text.append(clean_tweet)

    return np.array(clean_text)


def sentiment_analysis(clean_text):
    """
    Sentiment analysis for clean text according to Textblob library
    """

    analysis = TextBlob(clean_text)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    elif analysis.sentiment.polarity < 0:
        return -1


def analyze_text(clean_text):
    """
    Analyze the clean text tweet and classify them accordingly to positive, negative and neutral tweets.
    """

    data = pd.DataFrame(data=clean_text, columns=['Tweets'])

    data['Sentiment_analysis'] = np.array([sentiment_analysis(clean_text) for clean_text in data['Tweets'] ])

    pos_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if int(data['Sentiment_analysis'][index]) > 0]
    neutral_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if
                      int(data['Sentiment_analysis'][index]) == 0]
    neg_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if int(data['Sentiment_analysis'][index]) < 0]

    return data, pos_tweets, neg_tweets, neutral_tweets


def read_afinn_dict(filename):
    """
    Read afinn data
    """

    afinn = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                afinn[parts[0]] = int(parts[1])

    return afinn


def afinn_sentiment_analysis(afinn, clean_text):

    pos_senti = []
    neg_senti = []
    neutral_senti = []

    for tweet in clean_text:
        score = 0
        words = word_tokenize(tweet)
        for word in words:
            if word in afinn.keys():
                score += afinn[word]

        if score > 0:
            pos_senti.append(tweet)
        if score < 0:
            neg_senti.append(tweet)
        if score == 0:
            neutral_senti.append(tweet)

    return pos_senti, neg_senti, neutral_senti


def create_afinn_dataframe(pos_senti, neg_senti, neutral_senti):

    afinn_data = pd.DataFrame(columns=['Tweets', 'Sentiment_analysis'])

    for pos in pos_senti:
        afinn_data = afinn_data.append({'Tweets': pos, 'Sentiment_analysis': 1}, ignore_index=True)

    for neg in neg_senti:
        afinn_data = afinn_data.append({'Tweets': neg, 'Sentiment_analysis': -1}, ignore_index=True)

    for neut in neutral_senti:
        afinn_data = afinn_data.append({'Tweets': neut, 'Sentiment_analysis': 0}, ignore_index=True)

    afinn_data = afinn_data.sample(frac=1).reset_index(drop=True)
    return afinn_data


def write_file(data, positive, negative, neutral):

    f = open('classify_results.txt', 'a+', encoding='utf-8')

    f.write("\nNumber of Positive instances found: %d" % len(positive))
    f.write("\nNumber of Negative instances found : %d" % len(negative))
    f.write("\nNumber of Neutral instances found : %d\n" % len(neutral))

    f.write("\nPercentage of Positive tweets: {} %".format(len(positive) * 100 / len(data['Tweets'])))
    f.write("\nPercentage of Negative tweets: {} %".format(len(negative) * 100 / len(data['Tweets'])))
    f.write("\nPercentage of Neutral tweets: {} %".format(len(neutral) * 100 / len(data['Tweets'])))

    f.write("\n\nExample of a Positive class : %s" % positive[1])
    f.write("\nExample of a Negative class : %s" % negative[1])
    f.write("\nExample of a Neutral class : %s\n\n" % neutral[1])

    f.close()


def vectorize(data, vocab=None):

    if vocab is not None:
        count_vectorizer = CountVectorizer(binary='true', vocabulary=vocab)
    else:
        count_vectorizer = CountVectorizer(binary='true')

    tweets = data["Tweets"]
    vec_data = count_vectorizer.fit_transform(tweets)

    if vocab is None:
        vocab = count_vectorizer.vocabulary_

    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(vec_data)

    data["Vector"] = ""
    for index, vector in enumerate(tfidf_data):
        data["Vector"][index] = vector
    data['Vector'] = data['Vector'].astype(np.flatiter)
    return (data, vocab)

    pass


def unskew(train_data):

    pos = train_data.loc[train_data["Sentiment_analysis"] == 1, "Tweets"]
    neg = train_data.loc[train_data["Sentiment_analysis"] == -1, "Tweets"]
    neu = train_data.loc[train_data["Sentiment_analysis"] == 0, "Tweets"]

    for i in range(1, (int) (len(pos)/len(neg))):
        for tweet in neg:
            train_data = train_data.append({"Tweets": tweet, "Sentiment_analysis": -1}, ignore_index=True)

    for i in range(1, (int) (len(pos)/len(neu))):
        for tweet in neu:
            train_data = train_data.append({"Tweets": tweet, "Sentiment_analysis": 0}, ignore_index=True)

    return train_data


def df_append(train_data, test_data):

    train_data_x = []
    for vector in train_data['Vector']:
        train_data_x.append(tuple(vector.toarray()[0]))
    train_data_x = np.array(train_data_x)

    train_data_y = []
    for value in train_data["Sentiment_analysis"]:
        train_data_y.append(value)
    train_data_y = np.array(train_data_y)

    test_data_x = []
    for vector in test_data['Vector']:
        test_data_x.append(tuple(vector.toarray()[0]))
    test_data_x = np.array(test_data_x)

    test_data_y = []
    for value in test_data["Sentiment_analysis"]:
        test_data_y.append(value)
    test_data_y = np.array(test_data_y)

    return train_data_x, train_data_y, test_data_x, test_data_y


def main():

    print("Collecting data...............\n")
    tweets = load_data()
    print("Preprocessing/Cleaning the data ...............")
    clean_text = clean_data(tweets)

    # ------------------------- TEXTBLOB SENTIMENT ANALYSIS ON TWEETS PROCESSING ---------------------------------------
    print("\n********************* Reading data and analyzing the sentiment classification via TextBlob .......... \n")
    data, pos_tweets, neg_tweets, neutral_tweets = analyze_text(clean_text)

    f = open('classify_results.txt', 'w', encoding='utf-8')
    f.write("**************************Analyzing the tweets collected from collect file**************************\n")
    f.write("\nThe Number of Tweets collected : %d\n\n" % len(tweets))
    f.write("******************************************************************************************************")
    f.write("\n               SENTIMENT ANALYSIS VIA TEXTBLOB FOR TWEETS COLLECTED                               \n")
    f.write("*****************************************************************************************************\n")
    f.close()

    write_file(data, pos_tweets, neg_tweets, neutral_tweets)

    # ---------------------------------------- AFINN DATA PROCESS --------------------------------------------
    print("\n*********** Reading data and analyzing the sentiment classification via AFINN dict of words .......... \n")

    afinn = read_afinn_dict('words.txt')
    pos_senti, neg_senti, neutral_senti = afinn_sentiment_analysis(afinn, clean_text)

    f = open('classify_results.txt', 'a+', encoding='utf-8')
    f.write("******************************************************************************************************")
    f.write("\n               SENTIMENT ANALYSIS VIA AFINN DATASET FOR TWEETS COLLECTED                          \n")
    f.write("*****************************************************************************************************\n")
    f.close()

    write_file(data, pos_senti, neg_senti, neutral_senti)

    # --------------------------------------- LOGISTIC REGRESSION ON THE BASE MODEL - AFINN ---------------------------
    # Creating a dataframe to store the positive, negative and sentiment polarity along with tweets.
    print("\n*******************Building classifier using Count vectorizer, Logistic regression ********************\n")
    afinn_data = create_afinn_dataframe(pos_senti, neg_senti, neutral_senti)

    train_data, test_data = np.split(afinn_data, [int(.7 * len(afinn_data))])

    """
    Unskew the data according to the length of positive tweets retrieved from afinn_data above. Such that training data
    will have more or less equal number of positive, negative, neutral length.
    """
    train_data = unskew(train_data)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data, vocab = vectorize(train_data)
    test_data, vocab = vectorize(test_data, vocab)

    # ------------------ Append Vector values and sentiment analysis to their respective dataframes ------------------
    train_data_x, train_data_y, test_data_x, test_data_y = df_append(train_data, test_data)

    clf = LogisticRegression()
    clf.fit(train_data_x, train_data_y)

    pred = clf.predict(test_data_x)
    test_data["Predicted"] = ""
    for index, predicted in enumerate(pred):
        test_data["Predicted"][index] = predicted

    # ----------------------- Appending all the analysis output to classify_results.txt-----------------------------
    f = open('classify_results.txt', 'a+', encoding='utf-8')
    f.write("******************************************************************************************************")
    f.write("\n            CLASSIFIER BUILT ON AFINN DATA AND ITS LABELS USING LOGISTIC REGRESSION MODEL          \n")
    f.write("*****************************************************************************************************\n")
    f.write("The tweets are split into train(70%) and testing(30%) data and the scores are caluclated as below.\n")
    f.write("\n\nPositive tweets are high. Hence, for building classifier, the tweets in the training set are slightly "
            "oversampled to match the number of positive occurences.")
    f.write("\nThe score for trained data: %f " % clf.score(train_data_x, train_data_y))
    f.write("\nThe score for tested data: %f " % clf.score(test_data_x, test_data_y))

    f.write("\n\n****************** Trained data calculations below **********************************")
    f.write("\n\nTotal row count in trained data: %d" % train_data.shape[0])
    f.write("\n\nNumber of Positive Instances: %d " % len(train_data.loc[(train_data['Sentiment_analysis'] == 1)]))
    f.write("\nNumber of Negative Instances: %d " % len(train_data.loc[(train_data['Sentiment_analysis'] == 0)]))
    f.write("\nNumber of Neutral Instances: %d " % len(train_data.loc[(train_data['Sentiment_analysis'] == -1)]))
    f.write("\n\nExample of a Positive tweet: %s " % train_data.loc[train_data["Sentiment_analysis"] == 1, 'Tweets'].iloc[0])
    f.write("\nExample of a Negative tweet: %s " % train_data.loc[train_data["Sentiment_analysis"] == -1, 'Tweets'].iloc[0])
    f.write("\nExample of a Neutral Tweet: %s " % train_data.loc[train_data["Sentiment_analysis"] == 0, 'Tweets'].iloc[0])

    f.write("\n\n****************** Testing data calculations below **********************************")
    f.write("\n\nTotal row count in testing data: %d" % test_data.shape[0])
    if (len(test_data.loc[(test_data['Predicted'] == 1)]) or len(test_data.loc[(test_data['Predicted'] == -1)])
        or len(test_data.loc[(test_data['Predicted'] == 0)])) == 0:
        pass
    else:
        f.write("\n\nNumber of Positive Instances: %d " % len(test_data.loc[(test_data['Predicted'] == 1)]))
        f.write("\nNumber of Negative Instances: %d " % len(test_data.loc[(test_data['Predicted'] == 0)]))
        f.write("\nNumber of Neutral Instances: %d " % len(test_data.loc[(test_data['Predicted'] == -1)]))
    f.write("\n\nExample of a Positive tweet: %s " % test_data.loc[test_data["Predicted"] == 1, 'Tweets'].iloc[0])
    f.write("\nExample of a Negative tweet: %s " % test_data.loc[test_data["Predicted"] == -1, 'Tweets'].iloc[0])
    f.write("\nExample of a Neutral Tweet: %s " % test_data.loc[test_data["Predicted"] == 0, 'Tweets'].iloc[0])
    f.close()

    print("\nAll the results are captured in "'Classify.results.txt'" file \n")


if __name__ == '__main__':
    main()
