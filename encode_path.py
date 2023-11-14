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



# extract features from tag_path and content
def extract_features(path, content):
    tag = path[-1]
    
    # extract text length and punctuation count
    text = ''.join(content)
    text_length = len(text)
    punctuation_count = sum([1 for c in text if c in ['.', '!', '?', '。', '！', '？']])
    
    # extract class name
    classname = ''
    match = re.search(r'class="([^"]*)"', tag)
    if match:
        classname = match.group(1)
    tag = tag.split()[0]

    return np.array([tag, classname, text_length, punctuation_count])

# extract features from tag_path and content for encoding
# search for the encoding in tag_vectors and class_name_vectors first, if not found, get the encoding from the model
def encode_feature(path, content, model_tag, model_class_name, tag_vectors, class_name_vectors):
    tag, classname, text_length, punctuation_count = extract_features(path, content)
    # generate tag encoding

    try:
        tag_encoding = tag_vectors.loc[tag]
        tag_encoding = tag_encoding.vector
    except:
        tag_encoding = model_tag.wv[tag]

    # generate class name encoding
    if classname == '':
        classname_encoding = class_name_vectors.loc['0']
        classname_encoding = classname_encoding.vector
    else:
        try:
            classname_encoding = class_name_vectors.loc[classname]
            classname_encoding = classname_encoding.vector
        except:
            classname_encoding = model_class_name.wv[classname]


    # concatenate all features
    feature_vector = np.concatenate((tag_encoding, classname_encoding, np.array([float(text_length), float(punctuation_count)])))
    if len(feature_vector)<62:
        print("error")
    return feature_vector

# search the encoding of the parent node by recursion
# index: index of the current node
# tag_path_list: dom tree
# encoding_list: list of encoding of each node
def searchParentEncode(index, tag_path_list, encoding_list, model_tag, model_class_name, tag_vectors, class_name_vectors):
    path = tag_path_list[index]
    if len(path) < 2:
        return None
    parentTag = path[-2].split()[0]
    parent_path = path[:-2]
    j = index-1
    # found the parent node
    while j >= 0:
        tag = tag_path_list[j][-1].split()[0]
        if tag == parentTag and tag_path_list[j][:-1] == parent_path:
            break
        j -= 1

    if j>=0:
        return [j, encoding_list[j]]
    else:
        return None

def encode_path(index, content, model_tag, model_classname, tag_vectors, class_name_vectors, encoding_list, tag_path_list):
    # encode the current node
    encoding = encode_feature(tag_path_list[index], content, model_tag, model_classname, tag_vectors, class_name_vectors)
    # search for the encoding of the parent node
    parent = searchParentEncode(index, tag_path_list, encoding_list, model_tag, model_classname, tag_vectors, class_name_vectors)
    # return the encoding if the current node is the root node
    if parent is None:
        encoding_list[index] = [encoding]
        return encoding_list
    # found the parent node, if the parent node has encoding, append the encoding of the current node to the parent node's encoding
    elif not(parent[1] is None):
        encoding_parent = copy.copy(parent[1])
        encoding_parent.append(encoding)
        encoding_list[index] = encoding_parent
        return encoding_list
    # if the parent node does not have encoding, search for the encoding of the parent node's parent node by recursion
    else:
        encoding_list = encode_path(parent[0], content, model_tag, model_classname, tag_vectors, class_name_vectors, encoding_list, tag_path_list)
        encoding_parent = copy.copy(encoding_list[parent[0]])
        encoding_parent.append(encoding)
        encoding_list[index] = encoding_parent
        return encoding_list



# encode the path of each node
def encode_path_list(tag_path_list, content_list, index_non_empty_content, model_tag, model_classname, tag_vectors, class_name_vectors):
    encoding_list = [None] * len(tag_path_list)
    for index in index_non_empty_content:
        encoding_list = encode_path(index, content_list, model_tag, model_classname, tag_vectors, class_name_vectors, encoding_list, tag_path_list)
        # print(encoding_list[index])
    return encoding_list



