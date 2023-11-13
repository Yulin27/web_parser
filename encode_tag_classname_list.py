import csv
import os
from gensim.models.fasttext import FastText
from bs4 import BeautifulSoup
import pandas as pd

# 设置文件夹路径
folder_path = 'dataset/train_data/train_HTML'
model_file_tag = "encode/model_tag.bin"
model_file_class_name = "encode/model_class_name.bin"

# 加载HTML数据集
html_data = ''
for filename in os.listdir(folder_path):
    if filename.endswith('.html'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            html_data += f.read()

# 使用BeautifulSoup从HTML中提取标签和类名
soup = BeautifulSoup(html_data, 'html.parser')
tags = set()
class_names = set()
for tag in soup.find_all():
    tags.add(tag.name)
    if tag.has_attr('class'):
        # 将多个classname作为一个字符串添加到集合class_names中
        class_names.add(" ".join(tag['class']))

# 在提取的标签和类名上训练FastText模型
model_tag = FastText(vector_size=30, window=5, min_count=1, workers=4)
sentences = [list(tag) for tag in tags]
model_tag.build_vocab(corpus_iterable=sentences)
model_tag.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
# 模型保存
model_tag.save(model_file_tag)

model_class_name = FastText(vector_size=30, window=5, min_count=1, workers=4)
sentences = [list(class_name) for class_name in class_names]
model_class_name.build_vocab(corpus_iterable=sentences)
model_class_name.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=10)
model_class_name.save(model_file_class_name)

# 获取标签和类名的编码向量
tag_vectors = {}
class_name_vectors = {}
for tag in tags:
    if tag not in tag_vectors:
        encoding_tag = list(model_tag.wv[tag])
        # normalization
        max_value = max(encoding_tag)
        min_value = min(encoding_tag)
        encoding_tag = [(x - min_value) / (max_value - min_value) for x in encoding_tag]
        tag_vectors[tag] = encoding_tag

for class_name in class_names:
    if class_name not in class_name_vectors:
        encoding_class_name = list(model_class_name.wv[class_name])
        max_value = max(encoding_class_name)
        min_value = min(encoding_class_name)
        encoding_class_name = list(model_class_name.wv[class_name])
        encoding_class_name = [(x - min_value) / (max_value - min_value) for x in encoding_class_name]
        class_name_vectors[class_name] = encoding_class_name
class_name_vectors['0'] = list(model_class_name.wv[''])

# 转换为DataFrame
tag_vectors_df = pd.DataFrame(list(tag_vectors.items()), columns=['tag', 'vector'])
class_name_vectors_df = pd.DataFrame(list(class_name_vectors.items()), columns=['classname', 'vector'])

tag_vectors_df['vector'] = tag_vectors_df['vector'].astype(str)
class_name_vectors_df['vector'] = class_name_vectors_df['vector'].astype(str)

tag_vectors_df.set_index('tag', inplace=True)
class_name_vectors_df.set_index('classname', inplace=True)


tag_vectors_df.to_csv('encode/tag_vectors.csv', index=True)
class_name_vectors_df.to_csv('encode/class_name_vectors.csv', index=True)

