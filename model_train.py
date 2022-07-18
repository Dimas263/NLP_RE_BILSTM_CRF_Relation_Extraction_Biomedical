# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from att import Attention
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from operator import itemgetter

from load_data import get_train_test_pd
from extract_feature import BertVector

# Read the file and convert it
train_df, test_df = get_train_test_pd()
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=512)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]

train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)
print('end encoding')

# training set and test set
x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])
print('x_train: ', x_train.shape)

# Convert a value of type y to an ont-hot vector
num_classes = 4
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model structure: BERT + bidirectional GRU + Attention + FC
# inputs = Input(shape=(128, 768, ))
# gru = Bidirectional(GRU(128, dropout=0.2, return_sequences=True))(inputs)
# attention = Attention(32)(gru)
# output = Dense(num_classes, activation='softmax')(attention)
# model = Model(inputs, output)

# Model structure: BERT + bidirectional LSTM + Attention + FC
inputs = Input(shape=(512, 768, ))
bilstm = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(inputs)
attention = Attention(32)(bilstm)
output = Dense(num_classes, activation='softmax')(attention)
model = Model(inputs, output)

# Model visualization
from keras.utils import plot_model
plot_model(model, to_file='/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

# If there are .hdf5 files in the original models folder, delete them all
model_dir = '/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/output'
if os.listdir(model_dir):
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

# Save the latest val_acc best model file
filepath="/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/output/per-rel-{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='min')

# Model training and evaluation
history = model.fit(
  x_train, y_train, 
  validation_data=(x_test, y_test), 
  batch_size=16, 
  epochs=30, 
  callbacks=[early_stopping, checkpoint])
# model.save('people_relation.hdf5')

print('The effect on the test set：', model.evaluate(x_test, y_test))

# Read the relational correspondence table
with open('/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/input/rel_dict.json', 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())

sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]

# Output classification report for each class
y_pred = model.predict(x_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=values))

# draw loss and acc images
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("/content/drive/MyDrive/Rearch_Dimas/BILSTM_CRF_RE/output/loss_acc.png")