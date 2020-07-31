"""
This script loads the dataset build from the sources listed in sec. 4.1
Then it iterates over all movies that are contained in this dataset and also reads the domestic release data of that
movie (domrele). Using the search query described in sec. 4.1 twitter tweets are scraped using GetOldTweets3 API.
Then the fine-tuned BERT-Model is loaded and the tweets are classified on the continous scale [0,1]. In the end,
those predictions are aggregated on the movie level.
"""


import numpy as np
import tensorflow as tf
import GetOldTweets3 as got
import pandas as pd
from datetime import timedelta
from collections import defaultdict
import pickle
# import own module
from bert_load_and_predict import predict

# configs
MODEL_PATH = "../1594730309"
DATAMART_PATH = "../Data/final2.xlsx"
SAVE_SENT_PATH = '../Data/sent_scores.pkl'
SAVE_DF_SENT_PATH = '../Data/sent_scores_df.pkl'
SENT_FEATURES = ["mean_sent_score",
                 "median_sent_score",
                 "ratio_pos_tweets",
                 "total_tweets"]


def mean_sent_score(sent_preds):
    return np.mean(sent_preds)


def median_sent_score(sent_preds):
    return np.median(sent_preds)


def total_tweets(sent_preds):
    return len(sent_preds)


def ratio_pos_tweets(sent_preds):
    return len(predictions[sent_preds > 0.95]) / len(sent_preds)


def get_tweets_from_title(title, start_date, end_date, trailer=False):
    print("parsing tweets for {} from {} till {}".format(title, start_date, end_date))
    query = "{} AND (saw OR watched OR film OR movie OR trailer)".format(title)
    query = query + " AND trailer" if trailer else query
    tweet_criteria = got.manager.TweetCriteria().\
        setQuerySearch(query).setSince(start_date).\
        setUntil(end_date).setTopTweets(True).\
        setEmoji("unicode").setMaxTweets(1000)
    tweets = got.manager.TweetManager.getTweets(tweet_criteria)
    return [tweet.text for tweet in tweets]


if __name__ == "__main__":
    try:
        with open(SAVE_SENT_PATH, 'rb') as handle:
            sent_results = pickle.load(handle)
    except FileNotFoundError:
        sent_results = defaultdict(list)
    already_analysed_movies = list(sent_results.keys())
    df_mdb = pd.read_excel(DATAMART_PATH)
    df_mdb.columns = [x.lower().replace(" ", "_") for x in df_mdb.columns]
    df_mdb = df_mdb.loc[~df_mdb.title.isin(already_analysed_movies)]
    df_mdb.domrele = pd.to_datetime(df_mdb.domrele)
    # load BERT predictor
    predict_fn = tf.contrib.predictor.from_saved_model(MODEL_PATH)
    # sentiment analysis feature dictionary
    for i, movie in df_mdb.iterrows():
        print("Now doing sentiment analysis for movie: ", movie.title.lower())
        start_date = movie.domrele + timedelta(-17)
        end_date = start_date + timedelta(17)
        title = movie.title
        pred_sentences = get_tweets_from_title(str(title).lower(),
                                               str(start_date.date()),
                                               str(end_date.date()))
        if len(pred_sentences) == 0:
            continue
        print("get predictions")
        predictions = predict(pred_sentences[:], predict_fn)
        print("predictions successfully made")
        for feature in SENT_FEATURES:
            sent_results[title].append(eval(feature)(predictions))
        # the twitter api seems to be rather unstable, thus dump current results after each iteration
        with open(SAVE_SENT_PATH, 'wb') as handle:
            pickle.dump(sent_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # when the analysis is done, transform the dictionary to a pd Dataframe and dump it
    df_sent = pd.DataFrame.from_dict(sent_results, orient='index', columns=SENT_FEATURES)
    df_sent.to_pickle(SAVE_DF_SENT_PATH)
