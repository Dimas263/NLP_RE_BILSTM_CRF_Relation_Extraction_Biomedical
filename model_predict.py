# -*- coding: utf-8 -*-

import os, json
import numpy as np
from utils.extract_feature import BertVector
from keras.models import load_model
from utils.att import Attention

# Load the best trained model
model = load_model(
  "/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/results/uji 7-12/per-rel-11-0.7739.hdf5", 
  custom_objects={"Attention": Attention})

# Example Statements and Preprocessing
text1 = 'green tea#skin papillomas#Inhibitory effect of green tea on the growth of established skin papillomas in mice.'
per1, per2, doc = text1.split('#')
text = '$'.join([per1, per2, doc.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
print(text)


bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)
vec = bert_model.encode([text])["encodes"][0]
x_train = np.array([vec])

predicted = model.predict(x_train)
y = np.argmax(predicted[0])

with open('/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/input/rel_dict.json', 'r', encoding='utf-8') as f:
    rel_dict = json.load(f)

id_rel_dict = {v:k for k,v in rel_dict.items()}
print('original: %s' % text1)
print('Predict character relationships: %s' % id_rel_dict[y])
