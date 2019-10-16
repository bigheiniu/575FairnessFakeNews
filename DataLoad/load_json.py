import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=mp.cpu_count())
import pickle
from DataLoad.data_preprocess import clean_str, name_entity
import numpy as np
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
    news_df.loc[:, 'name_entity'] = news_df.loc[:, 'news'].parallel_apply(name_entity)
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
    data_path = ""
    save_path = ""

    load_news_content(data_path=data_path, save_path=save_path)




