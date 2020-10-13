# -*- coding: utf-8 -*
"""
author:lookingaf
使用深度学习LSTM进行imdb电影评论情感分析
"""
import numpy as np
import pandas

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, LSTM,  Bidirectional,GlobalMaxPooling1D,Dense, Input
from keras.models import  Model

MAX_NUM = 50000
SENTENCE_NUM = 25000
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000


def clean_text(string):
    string = string.decode('utf-8')
    string = string.lower().replace("<br />"," ")
    return string.strip().lower()

data = pandas.read_csv('./data.csv')#默认以逗号为分隔符

texts = [clean_text(text.encode('ascii','ignore')) for text in data.review]
labels = to_categorical(np.asarray(list(data.sentiment)))

embeddings_dict = {}
with open('data_50.txt','r',encoding='utf-8') as f:
    for data_word in f:
        word = data_word.split()[0]
        arra = np.asarray(data_word.split()[1:], dtype='float32')
        embeddings_dict[word] = arra

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

data = data[np.arange(MAX_NUM)]
labels = labels[np.arange(MAX_NUM)]

x_train = data[:24999]
y_train = labels[:24999]
x_test = data[25000:]
y_test = labels[25000:]

embedding_matrix = np.random.random((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            50,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

def Lstm():
    print('LSTM:')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')#输入层
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences)
    dense_1 = Dense(50,activation='sigmoid')(l_gru)
    dense_2 = Dense(2, activation='softmax')(dense_1)

    model = Model(sequence_input, dense_2)

    model.compile(loss='logcosh',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
               epochs=10,batch_size=50)

def MLP():
    print('MLP:')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # 输入层
    embedded_sequences = embedding_layer(sequence_input)
    # 通过计算用户和物品或物品和物品的Embedding相似度，来缩小推荐候选库的范围。实现高维稀疏特征向量向低维稠密特征向量的转换。训练好的embedding可以当作输入深度学习模型的特征。
    dense_1 = Dense(50, activation='relu')(embedded_sequences)  # 全连接神经网络层，隐藏层
    Max_pooling = GlobalMaxPooling1D()(dense_1)
    dense_2 = Dense(2, activation='softmax')(Max_pooling)  # 输出层

    model = Model(sequence_input, dense_2)
    model.compile(loss='logcosh',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=10, batch_size=50)

Lstm()
MLP()