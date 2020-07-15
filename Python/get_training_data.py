import os
import re
import pandas as pd
import tensorflow as tf
import pytreebank
from collections import defaultdict


def get_stanford_sent_data(kind="train"):
    dataset = pytreebank.load_sst()
    labeled_sentences = defaultdict(list)
    for phrase in range(len(dataset[kind])):
        example = dataset[kind][phrase]
        label, sentence = example.to_labeled_lines()[0]
        labeled_sentences["sentiment"].append(label)
        labeled_sentences["sentence"].append(sentence)
    df = pd.DataFrame.from_dict(labeled_sentences)
    return df


def get_and_process_stanford_sent_data(kind="train"):
    sf_df = get_stanford_sent_data(kind)
    # dont bias the model with neutral sentences during training stage
    neg_df = sf_df[sf_df.sentiment <= 1]
    pos_df = sf_df[sf_df.sentiment >= 3]
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def sf_download_and_load_dataset():
    train_df = get_and_process_stanford_sent_data(kind="train")
    test_df = get_and_process_stanford_sent_data(kind="test")
    return train_df, test_df


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    # dont bias the model with neutral sentences during training stage
    train_df = train_df[(train_df.sentiment.astype('int') <= 3) | (train_df.sentiment.astype('int') >= 8)]
    return train_df, test_df


def get_full_train_and_test():
    imdb_df_train, imdb_df_test = download_and_load_datasets()
    sf_df_train, sf_df_test = sf_download_and_load_dataset()
    df_train = pd.concat([imdb_df_train, sf_df_train], axis=0).reset_index(drop=True)
    df_test = pd.concat([imdb_df_test, sf_df_test], axis=0).reset_index(drop=True)
    return df_train, df_test

if __name__ == "__main__":
    a, b = get_full_train_and_test()
    pd.to_pickle(a, "../Data/train.pkl")
    pd.to_pickle(b, "../Data/test.pkl")