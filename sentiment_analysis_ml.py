# -*- coding: utf-8 -*
"""
author:lookingaf
使用机器学习进行imdb电影评论情感分析
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import pandas
from os.path import join
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


paths = './aclImdb'

def clean_text(string):
    string = string.decode('utf-8')
    string = string.lower().replace("<br />"," ")
    return string.strip().lower()

def preprocessing():
    data = pandas.DataFrame()
    labels = {'pos':1,'neg':0}
    for path in ['train','test']:
        for label in labels.keys():
            label_path = '{}'.format(paths + '/' + path + '/' + label)
            for file in os.listdir(label_path):
                with open(join(label_path, file), 'r', encoding='utf-8') as f:
                    review = f.read()
                    review = clean_text(review)
                    data = data.append([[labels[label],review]])
    data.columns = ['sentiment', 'review']
    return data

#data = preprocessing()
#data.to_csv('./data.csv',index=None)
data = pandas.read_csv('./data.csv')#默认以逗号为分隔符
x_orig = data.review
y_orig = data.sentiment

import numpy as np
cv = CountVectorizer(max_features=1000)#CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数。CountVectorizer当然适用线性模型，对前300个词进行降序排序
#停用词默认为英语：停用词是指在信息检索中，为节省存储空间和提高搜索效率，在处理自然语言数据（或文本）之前或之后会自动过滤掉某些字或词，这些字或词即被称为Stop Words（停用词）
cv.fit(x_orig)#Learn a vocabulary dictionary of all tokens in the raw documents.
train_data = cv.transform(x_orig)#ransform documents to document-term matrix.
train_data = train_data.toarray()#将矩阵转化为数组

tfidf = TfidfVectorizer(max_features=1000)#TfidfVectorizer对词项用idf值进行改进，也就是考虑了词项在文档间的分布，也适用于线性模型，同时由于通常线性模型要求输入向量的模为1，因此TfidfVectorizer默认行向量是单位化后的。_idf_diag为一个词在全局的比重
tfidf.fit(x_orig)#统计词频
tfidf_train_data = tfidf.transform(x_orig)#进行转换成矩阵
# (the index of the list , the index of the dict ) the frequency of the list[index]
tfidf_train_data = tfidf_train_data.toarray()

def train(x):
    y = y_orig
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    # 训练数据
    module = MultinomialNB()  # 离散型朴素贝叶斯—— MultinomialNB
    module.fit(x_train, y_train)  # 训练
    # 测试数据
    y_pred = module.predict(x_test)
    # 输出
    print("正确值：{0}".format(y_test))
    print("预测值：{0}".format(y_pred))
    print("准确率：%f%%" % (accuracy_score(y_test, y_pred) * 100))

print('TfidfVectorizer:')
train(tfidf_train_data)
print('\nCountVectorizer:')
train(train_data)
