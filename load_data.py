# -*- coding: utf-8 -*-
import pandas as pd


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines()]

    labels, texts = [], []
    for line in content:
        parts = line.split()
        label, text = parts[0], ''.join(parts[1:])
        labels.append(label)
        texts.append(text)

    return labels, texts


def get_train_test_pd():
    file_path = '/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/input/train.txt'
    labels, texts = read_txt_file(file_path)
    train_df = pd.DataFrame({'label': labels, 'text': texts})

    file_path = '/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/input/test.txt'
    labels, texts = read_txt_file(file_path)
    test_df = pd.DataFrame({'label': labels, 'text': texts})

    return train_df, test_df


if __name__ == '__main__':

    train_df, test_df = get_train_test_pd()
    print(train_df.head())
    print(test_df.head())

    train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
    print(train_df.describe())
