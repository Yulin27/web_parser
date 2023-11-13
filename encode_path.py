from preprocess import *
from gensim.models.fasttext import FastText
import numpy as np
import time
import os
import copy
import pandas as pd


model_params = {
    'vector_size': 30,
    'min_count': 1,
}

def read_csv(file_name):
    df = pd.read_csv(file_name)
    for i in range(len(df)):
        df.iloc[i]['vector'] = eval(df.iloc[i]['vector'])
    return df



# 定义函数进行特征提取
def extract_features(path, content):
    # 提取标签
    tag = path[-1]
    
    # 提取文本长度
    # 各种语言的长度计算方式不同，这里只是简单的计算字符数
    text = ''.join(content)
    text_length = len(text)
    # 提取包含的结束标点符号数
    punctuation_count = sum([1 for c in text if c in ['.', '!', '?', '。', '！', '？']])
    
    # 提取classname信息
    classname = ''
    match = re.search(r'class="([^"]*)"', tag)
    if match:
        classname = match.group(1)
    tag = tag.split()[0]

    return np.array([tag, classname, text_length, punctuation_count])

# 定义函数进行特征编码
# 优先在tag_vectors和class_name_vectors中查找编码，如果没有找到，再从模型中获取编码
def encode_feature(path, content, model_tag, model_class_name, tag_vectors, class_name_vectors):
    tag, classname, text_length, punctuation_count = extract_features(path, content)
    # 生成标签编码

    try:
        tag_encoding = tag_vectors.loc[tag]
        tag_encoding = tag_encoding.vector
    except:
        tag_encoding = model_tag.wv[tag]

    # 生成类名编码
    if classname == '':
        classname_encoding = class_name_vectors.loc['0']
        classname_encoding = classname_encoding.vector
    else:
        try:
            classname_encoding = class_name_vectors.loc[classname]
            classname_encoding = classname_encoding.vector
        except:
            classname_encoding = model_class_name.wv[classname]


    # 生成输出特征编码
    feature_vector = np.concatenate((tag_encoding, classname_encoding, np.array([float(text_length), float(punctuation_count)])))
    if len(feature_vector)<62:
        print("error")
    return feature_vector

# 递归查找父节点的编码
# index: 当前节点的索引
# tag_path_list: dom树
# encoding_list: 编码列表
# 如果找到的父节点有内容，父节点的编码应该已经存储在encoding_list中，直接返回
def searchParentEncode(index, tag_path_list, encoding_list, model_tag, model_class_name, tag_vectors, class_name_vectors):
    path = tag_path_list[index]
    if len(path) < 2:
        return None
    parentTag = path[-2].split()[0]
    parent_path = path[:-2]
    j = index-1
    # 找到父节点
    while j >= 0:
        tag = tag_path_list[j][-1].split()[0]
        if tag == parentTag and tag_path_list[j][:-1] == parent_path:
            break
        j -= 1

    if j>=0:
        return [j, encoding_list[j]]
    else:
        return None

# 递归编码路径并返回encoding_list
def encode_path(index, content, model_tag, model_classname, tag_vectors, class_name_vectors, encoding_list, tag_path_list):
    # 编码当前节点
    encoding = encode_feature(tag_path_list[index], content, model_tag, model_classname, tag_vectors, class_name_vectors)
    # 查找父节点的位置和编码
    parent = searchParentEncode(index, tag_path_list, encoding_list, model_tag, model_classname, tag_vectors, class_name_vectors)
    # html的根节点没有父节点，直接返回
    if parent is None:
        encoding_list[index] = [encoding]
        return encoding_list
    # 找到父节点，如果父节点有编码,将当前节点的编码加入到父节点的编码中,作为当前节点的编码路径
    elif not(parent[1] is None):
        encoding_parent = copy.copy(parent[1])
        encoding_parent.append(encoding)
        encoding_list[index] = encoding_parent
        return encoding_list
    # 找到父节点，如果父节点没有编码，递归查找父节点的父节点
    else:
        encoding_list = encode_path(parent[0], content, model_tag, model_classname, tag_vectors, class_name_vectors, encoding_list, tag_path_list)
        encoding_parent = copy.copy(encoding_list[parent[0]])
        encoding_parent.append(encoding)
        encoding_list[index] = encoding_parent
        return encoding_list



# 编码所有路径
def encode_path_list(tag_path_list, content_list, index_non_empty_content, model_tag, model_classname, tag_vectors, class_name_vectors):
    encoding_list = [None] * len(tag_path_list)
    for index in index_non_empty_content:
        encoding_list = encode_path(index, content_list, model_tag, model_classname, tag_vectors, class_name_vectors, encoding_list, tag_path_list)
        # print(encoding_list[index])
    return encoding_list



