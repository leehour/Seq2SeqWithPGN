import pandas as pd
import os
import jieba
import re
from config import train_path, test_path, train_seg_path, test_seg_path, stop_words_path, train_seg_merge_path, \
    test_seg_merge_path


# stop_words
def get_stop_words(stop_words_path):
    stop_words = '，：？。? ！! @ # $ % ^ & * ( （ ） ) [ ] { } > < = - + ~ ` --- (i (or / ; ;\' $1 |> \
                   0 1 2 3 4 5 6 7 8 9 13 15 30 24 20 "a" tk> 95 45'
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = re.sub("\n", " ", line)
            stop_words = line + stop_words
    return stop_words


def process(s):
    stop_words = get_stop_words(stop_words_path)
    seg = [i for i in jieba.cut(s) if i not in stop_words]
    return " ".join(seg)


def build_vocab(df, sort=True, min_count=0, lower=False):
    data_columns = df.columns.tolist()
    df_new = pd.DataFrame()
    for col in data_columns:
        df[col] = df[col].apply(str)
        df_new[col] = df[col].apply(process)
    return df_new


if __name__ == '__main__':
    df_train = pd.read_csv(train_path, encoding='utf-8')
    df_test = pd.read_csv(test_path, encoding='utf-8')

    # split words
    df_train_split = build_vocab(df_train)
    df_test_split = build_vocab(df_test)

    # drop lines contains null
    df_train_split.dropna(axis=0, how='any', inplace=True)
    df_test_split.dropna(axis=0, how='any', inplace=True)

    # merge columns except the Report
    df_train_split['input'] = df_train_split['Brand'] + ' ' + df_train_split['Model'] + ' ' + df_train_split[
        'Question'] + ' ' + df_train_split['Dialogue']
    df_train_split.drop(['Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)

    df_test_split['input'] = df_test_split['Brand'] + ' ' + df_test_split['Model'] + ' ' + df_test_split[
        'Question'] + ' ' + df_test_split[
                                 'Dialogue']
    df_test_split.drop(['Brand', 'Model', 'Question', 'Dialogue'], axis=1, inplace=True)

    df_train_split.to_csv(train_seg_merge_path, index=False)
    df_test_split.to_csv(test_seg_merge_path, index=False)
