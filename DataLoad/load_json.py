import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=mp.cpu_count())
import pickle
# from DataLoad.data_preprocess import clean_str, name_entity
import numpy as np
import re
import spacy
from spacy.lang.en import English

import re
nlp = English()
name_entity_nlp = spacy.load('en_core_web_sm')

# MAX_CHARS = 20000
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
              'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
              'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
              'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
              'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
              'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
              'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
              'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
              'shouldn', 'wasn', 'weren', 'won', 'wouldn']


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = p.clean(string)
    string = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ",
        str(string))
    string = re.sub(r"[ ]+", " ", string)
    string = re.sub(r"\!+", "!", string)
    string = re.sub(r"\,+", ",", string)
    string = re.sub(r"\?+", "?", string)

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string_list = [x.text for x in nlp.tokenizer(string) if x.text != " "]
    # remove stop words
    string_list = [word for word in string_list if word not in stop_words]
    return " ".join(string_list)

def name_entity(news_content, type="PERSON"):
    docs = nlp(news_content)
    ents = [e.text for e in docs.ents if e.label_ == type]
    print(docs)
    print(ents)
    exit()
    return ents
def load_news_content(data_path, save_path):
    real_news_comments = []
    fake_news_comments = []
    for direc in os.listdir(data_path):
        direc = os.path.join(data_path, direc)
        if "_fake" in direc:
            for file in tqdm(os.listdir(direc)):
                with open(os.path.join(direc, file, "news_article.json"), "r") as f1:
                    data = json.load(f1)
                    try:
                        news_fake = data["text"].replace("\n", "")
                    except:
                        print(os.path.join(direc, file, "news_article.json"))
                        continue
                fake_news_comments.append((news_fake, 0))

        elif "_real" in direc:
            for file in tqdm(os.listdir(direc)):
                with open(os.path.join(direc, file, "news_article.json"), "r") as f1:
                    data = json.load(f1)
                    try:
                        news_real = data["text"].replace("\n", "")
                    except:
                        print(os.path.join(direc, file, "news_article.json"))
                        continue

                real_news_comments.append((news_real, 1))

    news_df = pd.DataFrame(real_news_comments + fake_news_comments, columns=['news', 'label'])
    news_df['id'] = news_df.index
    news_df.loc[:, 'news'] = news_df.loc[:, "news"].parallel_apply(clean_str)
    news_df.to_csv("{}/news_label.csv".format(save_path), index=None)
    return news_df

def data_df_name_entity(news_df, save_path):
    news_df.loc[:, 'name_entity'] = news_df.loc[:, 'news'].apply(name_entity)
    news_df.to_csv("{}/news_label_name.csv".format(save_path))
    return news_df


def train_test_val_split(news_df, save_path):
    train_ratio = 0.55
    val_ratio = 0.2
    test_ratio = 0.25
    np.random.seed(123)
    index = np.array(list(range(len(news_df))))
    index_length = len(index)
    np.random.shuffle(index)
    train_index = index[:int(train_ratio * index_length)]
    test_index = index[len(train_index): int((train_ratio + test_ratio) * index_length)]
    val_index = index[len(train_index) + len(test_index):]
    train_data = news_df.loc[train_index, :]
    test_data = news_df.loc[test_index, :]
    val_data = news_df.loc[val_index, :]
    train_data.to_csv("{}/train.csv".format(save_path), index=None)
    test_data.to_csv("{}/test.csv".format(save_path), index=None)
    val_data.to_csv("{}/val.csv".format(save_path), index=None)
    return train_data, test_data, val_data


if __name__ == '__main__':
    data_path = "/home/yichuan/fake_news_data/political_fact"
    save_path = "/home/yichuan/fair_trump/data/political"

    # df = load_news_content(data_path=data_path, save_path=save_path)
    df = pd.read_csv(save_path + "/news_label_name.csv")
    df.dropna(how='any', inplace=True)
    data_df_name_entity(df, save_path)




