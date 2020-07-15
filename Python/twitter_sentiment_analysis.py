import numpy as np
import tensorflow as tf
import GetOldTweets3 as got
import pandas as pd

# import own module
from bert_load_and_predict import predict

# configs
MODEL_PATH = "../1594730309"


def get_tweets_from_title(title, start_date, end_date, trailer=False):
    query = "{} AND (saw OR watched)".format(title)
    query = query + " AND trailer" if trailer else query
    tweetCriteria = got.manager.TweetCriteria().\
        setQuerySearch(query).setSince(start_date).\
        setUntil(end_date).setTopTweets(True).\
        setEmoji("unicode").setMaxTweets(100)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    return [tweet.text for tweet in tweets]

start_date = "2016-12-15"
end_date = "2016-12-28"
if __name__ == "__main__":
    predict_fn = tf.contrib.predictor.from_saved_model(MODEL_PATH)
    pred_sentences = get_tweets_from_title("Rogue One: A Star Wars Story", start_date, end_date)
    predictions = predict(pred_sentences[:], predict_fn)
    print("\n".join([pred_sentences[i] + ": " + str(predictions[i]) for i in range(len(predictions))]))
    print("%f : mean sentiment score" % np.mean(predictions))
    print("%f very positive tweets" % len(predictions[predictions > 0.95]))
    print("%f tweets in total" % len(predictions))