# -*- coding: utf-8 -*-
import json
import pandas as pd
from pprint import pprint

df = pd.read_excel('ori_data/gold-standard-corpus.xlsx')
relations = list(df['relation'].unique())
relations.remove('Negative')
relation_dict = {'Negative': 0}
relation_dict.update(dict(zip(relations, range(1, len(relations)+1))))

with open('rel_dict.json', 'w', encoding='utf-8') as h:
    h.write(json.dumps(relation_dict, ensure_ascii=False, indent=2))

print('total: %s' % len(df))
pprint(df['relation'].value_counts())
df['rel'] = df['relation'].apply(lambda x: relation_dict[x])

texts = []
for per1, per2, text in zip(df['plant'].tolist(), df['disease'].tolist(), df['sentence'].tolist()):
    text = '$'.join([per1, per2, text.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
    texts.append(text)

df['text'] = texts

# df = df.iloc[:100, :]

train_df = df.sample(frac=0.8, random_state=1024)
test_df = df.drop(train_df.index)

with open('train.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(train_df['text'].tolist(), train_df['rel'].tolist()):
        f.write(str(rel)+' '+text+'\n')

with open('test.txt', 'w', encoding='utf-8') as g:
    for text, rel in zip(test_df['text'].tolist(), test_df['rel'].tolist()):
        g.write(str(rel)+' '+text+'\n')
