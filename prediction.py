import torch
import torch.nn as nn
from preprocess import *
import torch
import torch.nn as nn
import numpy as np
import config
from encode_path import *
import sys
import getopt
import pickle




tag_vectors = read_csv(config.tag_vectors_file)
class_name_vectors = read_csv(config.class_name_vectors_file)
tag_vectors.set_index('tag', inplace=True)
class_name_vectors.set_index('classname', inplace=True)

model_tag = FastText.load(config.model_file_tag)
model_classname = FastText.load(config.model_file_class_name)



def get_data(file_content, filename):
    html_content = preprocess_html_2(file_content, filename)
    paths = dom_tree(html_content)

    content_list = assign_content_to_tags(paths, html_content)

    non_empty_index = extract_nonempty_content_index(content_list)
    nonempty_content = [content_list[i] for i in non_empty_index]

    encoding_list = encode_path_list(paths, content_list, non_empty_index, model_tag, model_classname, tag_vectors, class_name_vectors)
    encoding_list = [encoding_list[i] for i in non_empty_index]

    # padding and truncate
    for i in range(len(encoding_list)):
        if len(encoding_list[i]) > 15:
            encoding_list[i] = encoding_list[i][:15]
    try:
        max_length = max(len(path) for path in encoding_list)
    except:
        print("Cannot analyze this file: ","\nPlease check if the page is accessible")
        return None, None
    tag_path_padded = np.array([path + [[0.]*62] * (max_length - len(path)) for path in encoding_list])
    data = torch.tensor(tag_path_padded, dtype=torch.float)
    return data, nonempty_content
    

def load_normalization_params(normalization_params_path):
    with open(normalization_params_path, "rb") as f:
        normalization_params = pickle.load(f)
    return normalization_params


def normalize(X, normalization_params):
    x = X.numpy()
    x += 1e-16

    label_encoding = x[:, :, :30]
    attribute_encoding = x[:, :, 30:60]
    text_length = x[:, :, 60]
    punctuation_nb = x[:, :, 61]

    # Min-max normalization
    label_min = np.min(label_encoding, axis=2)
    label_max = np.max(label_encoding, axis=2)
    normalized_label_encoding = (label_encoding - label_min[:, :, np.newaxis]) / ((label_max[:, :, np.newaxis] - label_min[:, :, np.newaxis])+1e-16)

    
    attribute_min = np.min(attribute_encoding, axis=2)
    attribute_max = np.max(attribute_encoding, axis=2)
    normalized_attribute_encoding = (attribute_encoding - attribute_min[:, :, np.newaxis]) / ((attribute_max[:, :, np.newaxis] - attribute_min[:, :, np.newaxis])+1e-16)

    
    # Z-score normalization
    text_length_mean = normalization_params["text_length_mean"]
    text_length_std = normalization_params["text_length_std"]
    normalized_text_length = (text_length - text_length_mean) / text_length_std
    normalized_text_length = normalized_text_length[:, :, np.newaxis]

    punctuation_nb_mean = normalization_params["punctuation_nb_mean"]
    punctuation_nb_std = normalization_params["punctuation_nb_std"]
    normalized_punctuation_nb = (punctuation_nb - punctuation_nb_mean) / punctuation_nb_std
    normalized_punctuation_nb = normalized_punctuation_nb[:, :, np.newaxis]

    
    normalized_x = np.concatenate((normalized_label_encoding, normalized_attribute_encoding, normalized_text_length, normalized_punctuation_nb), axis=2)
    return normalized_x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  
        output = self.fc(lstm_out)
        return output

def filter(test_preds, content):
    for i in range(len(test_preds)):
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in content[i])
        if not is_chinese:
            if len(content[i]) <= 3:
                test_preds[i] = 0
            if len(content[i]) > 300:
                test_preds[i] = 1
        else:
            if len(content[i]) <= 3:
                test_preds[i] = 0
            if len(content[i]) > 100:
                test_preds[i] = 1
    return test_preds

def load_model(model_file, model):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model



def prediction(file_html,filename, normalization_params, model):
    X_test, content = get_data(file_html,filename)
    if X_test is None:
        return
    X_test_norm = normalize(X_test,normalization_params)
    X_test_norm = torch.from_numpy(X_test_norm).float()
    test_preds = []
    prediction = model(X_test_norm)
    test_preds.extend(torch.sigmoid(prediction).cpu().detach().numpy().tolist())
    test_preds = np.array(test_preds)
    test_preds = (test_preds > 0.5).astype(int)
    test_preds = filter(test_preds, content)

    return_content = ""
    for i in range(len(test_preds)):
        if test_preds[i]:
            return_content += content[i]
    
    return_content = re.sub('<[^>]+>', '', return_content)

    return return_content



def main(argv):
    model_type = "zh"  # default model type
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:", ["ifile=", "ofile=", "model="])
    except getopt.GetoptError:
        print('prediction.py -i <inputfile> -o <outputfile> -m <modeltype>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile> -m <modeltype>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--model"):
            model_type = arg
    print('Input File：', inputfile)
    print('Output File：', outputfile)
    with open(inputfile, "r", encoding="utf-8") as f:
        html = f.read()
        file = inputfile
        file_name = file.split("/")[-1]
        file_name_pur = file_name.split(".")[0]

        if model_type == "zh":
            model_path = "model/model_LSTM_zh.pth"
            normalization_params = load_normalization_params("model/normalization_params_zh.pkl")

        elif model_type == "la":
            model_path = "model/model_LSTM_la.pth"
            normalization_params = load_normalization_params("model/normalization_params_la.pkl")
        else:
            print("Model Invalid！")
            sys.exit()

        model = LSTMModel(config.input_size, config.hidden_size, config.num_layers, config.output_size, config.dropout)
        model = load_model(model_path, model)

        return_content = prediction(html, file_name_pur, normalization_params, model)
        with open(outputfile, "w", encoding="utf-8") as f:
            f.write(return_content)

if __name__ == "__main__":
    main(sys.argv[1:])

