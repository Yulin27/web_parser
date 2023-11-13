import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from labeliser import *
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
import config
from sklearn.model_selection import train_test_split
from encode_path import *
import pickle



tag_vectors = read_csv(config.tag_vectors_file)
class_name_vectors = read_csv(config.class_name_vectors_file)
tag_vectors.set_index('tag', inplace=True)
class_name_vectors.set_index('classname', inplace=True)

model_tag = FastText.load(config.model_file_tag)
model_classname = FastText.load(config.model_file_class_name)



def get_data(html_file_path):
    # 进行处理，将content内容写入txt文件
    with open(html_file_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

        html_content = preprocess_html(html_content)
        # 调用函数生成标签路径
        paths = dom_tree(html_content)
        # 调用函数生成对应内容表
        content_list = assign_content_to_tags(paths, html_content)

        # 调用函数提取非空内容的索引
        non_empty_index = extract_nonempty_content_index(content_list)
        nonempty_content = [content_list[i] for i in non_empty_index]

        encoding_list = encode_path_list(paths, content_list, non_empty_index, model_tag, model_classname, tag_vectors, class_name_vectors)
        encoding_list = [encoding_list[i] for i in non_empty_index]
        # 最大序列长度 15 长于15个的截断，短于15个的补0）
        for i in range(len(encoding_list)):
            if len(encoding_list[i]) > 15:
                encoding_list[i] = encoding_list[i][:15]

        max_length = max(len(path) for path in encoding_list)
        tag_path_padded = np.array([path + [[0.]*62] * (max_length - len(path)) for path in encoding_list])
        data = torch.tensor(tag_path_padded, dtype=torch.float)
        return data, nonempty_content
    
def get_data_tag(html_file_path, text_df):
    # 进行处理，将content内容写入txt文件
    with open(html_file_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
        file_name = html_file_path.split('/')[-1]
        file_name_pur = os.path.splitext(file_name)[0]

        html_content = preprocess_html(html_content)
        # 调用函数生成标签路径
        paths = dom_tree(html_content)
        # 调用函数生成对应内容表
        content_list = assign_content_to_tags(paths, html_content)

        # 调用函数提取非空内容的索引
        non_empty_index = extract_nonempty_content_index(content_list)
        nonempty_content = [content_list[i] for i in non_empty_index]


        encoding_list = encode_path_list(paths, content_list, non_empty_index, model_tag, model_classname, tag_vectors, class_name_vectors)
        encoding_list = [encoding_list[i] for i in non_empty_index]
        # 最大序列长度 15 长于15个的截断，短于15个的补0）
        for i in range(len(encoding_list)):
            if len(encoding_list[i]) > 15:
                encoding_list[i] = encoding_list[i][:15]


        text = text_df[text_df['Filename'] == file_name_pur]['Body text'].values[0]
        target = generate_target(nonempty_content, text)
        return encoding_list, target, nonempty_content
        
    

def get_data_tags(html_floder, text_file, isTest=False):
    # 将标签路径信息和相应的标签信息转换为数值表示的数据作为输入数据
    
    tag_path_idx = []
    tag_idx = []
    content_all = []

    # 遍历源文件夹下的所有文件
    with open(text_file, 'r', encoding='utf-8') as file_text:
        lines = file_text.readlines()

    # 创建空的 DataFrame
    df = pd.DataFrame(columns=["Filename", "Body text", "Website"])

    # 解析每一行文本并添加到 DataFrame 中
    for line in lines:
        if line.startswith("{"):
            row = eval(line)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    for filename in os.listdir(html_floder):
        print(filename)
        if filename.endswith(".html"):
            # 构造html文件的完整路径
            html_file_path = os.path.join(html_floder, filename)
            
            encoding_list, target, content = get_data_tag(html_file_path, df)
            content_all.extend(content)
            tag_path_idx.extend(encoding_list)
            tag_idx.extend(target)

    # 零填充
    max_length = max(len(path) for path in tag_path_idx)
    tag_path_padded = np.array([path + [[0.]*62] * (max_length - len(path)) for path in tag_path_idx])
    tag_idx = np.array(tag_idx)

    data = torch.tensor(tag_path_padded, dtype=torch.float)
    targets = torch.tensor(tag_idx, dtype=torch.int)
    if isTest:
        return data, targets, content_all
    return data, targets
   
   
def normalize(x, params_dict):
    if type(x)==torch.Tensor:
        x = x.numpy()
    x += 1e-16
    label_encoding = x[:, :, :30]
    attribute_encoding = x[:, :, 30:60]
    text_length = x[:, :, 60]
    punctuation_nb = x[:, :, 61]

    # 最大最小值归一化
    label_min = np.min(label_encoding, axis=2)
    label_max = np.max(label_encoding, axis=2)
    normalized_label_encoding = (label_encoding - label_min[:, :, np.newaxis]) / ((label_max[:, :, np.newaxis] - label_min[:, :, np.newaxis])+1e-16)

    
    attribute_min = np.min(attribute_encoding, axis=2)
    attribute_max = np.max(attribute_encoding, axis=2)
    normalized_attribute_encoding = (attribute_encoding - attribute_min[:, :, np.newaxis]) / ((attribute_max[:, :, np.newaxis] - attribute_min[:, :, np.newaxis])+1e-16)

    
    # Z-score标准化
    if params_dict == {}:
        text_length_mean = np.mean(text_length)
        text_length_std = np.std(text_length)
        params_dict['text_length_mean'] = text_length_mean
        params_dict['text_length_std'] = text_length_std
        punctuation_nb_mean = np.mean(punctuation_nb)
        punctuation_nb_std = np.std(punctuation_nb)
        params_dict['punctuation_nb_mean'] = punctuation_nb_mean
        params_dict['punctuation_nb_std'] = punctuation_nb_std
    else:
        text_length_mean = params_dict['text_length_mean']
        text_length_std = params_dict['text_length_std']
        punctuation_nb_mean = params_dict['punctuation_nb_mean']
        punctuation_nb_std = params_dict['punctuation_nb_std']

    normalized_text_length = (text_length - text_length_mean) / text_length_std
    normalized_text_length = normalized_text_length[:, :, np.newaxis]

    normalized_punctuation_nb = (punctuation_nb - punctuation_nb_mean) / punctuation_nb_std
    normalized_punctuation_nb = normalized_punctuation_nb[:, :, np.newaxis]

    
    # 将标签编码、属性编码、文本长度和标点符号数量拼接在一起
    normalized_x = np.concatenate((normalized_label_encoding, normalized_attribute_encoding, normalized_text_length, normalized_punctuation_nb), axis=2)
    return normalized_x         



def save_normalization_params(normalization_params_path, text_length_mean, text_length_std, punctuation_nb_mean, punctuation_nb_std):
    normalization_params = {

        "text_length_mean": text_length_mean,
        "text_length_std": text_length_std,
        "punctuation_nb_mean": punctuation_nb_mean,
        "punctuation_nb_std": punctuation_nb_std
    }
    with open(normalization_params_path, "wb") as f:
        pickle.dump(normalization_params, f)
        
def load_normalization_params(normalization_params_path):
    with open(normalization_params_path, "rb") as f:
        normalization_params = pickle.load(f)
    return normalization_params


def calculate_f1_score(y_true, y_pred):
    # 将标签转换为二进制形式
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred)

    return f1


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        output = self.fc(lstm_out)
        return output


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 注意这里的 hidden_size 乘以 2

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        output = self.fc(lstm_out)
        return output


# 修改 MyDataset 类
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index].clone().detach().requires_grad_(True)
        y = self.y[index].clone().detach().float().requires_grad_(True)
        return x, y


def load_model(model_file, model):
    # 加载保存的权重
    model.load_state_dict(torch.load(model_file))
    # 设置模型为评估模式（如果需要进行推理）
    model.eval()
    return model
    

def main():
    X_train, y_train = get_data_tags(config.train_html, config.train_text)

    # X_test, y_test, content_all = get_data_tags(config.test_html, config.test_text, isTest=True)

    params_dict = {}
    normalized_X_train = normalize(X_train, params_dict)


    # 调用函数保存归一化参数
    # save_normalization_params("normalization_params_la.pkl", params_dict['text_length_mean'], # # params_dict['text_length_std'], params_dict['punctuation_nb_mean'], #params_dict['punctuation_nb_std'])

    # Set hyperparameters
    dropout = config.dropout
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    output_size = config.output_size
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    patience = config.patience
    input_size = config.input_size


    # validation croisee
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    res = []
    Y_train_copy = copy.deepcopy(y_train)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(normalized_X_train, Y_train_copy)):
        print(f'Fold {fold + 1}')
        res_sub = []
        X_train, X_val = normalized_X_train[train_idx], normalized_X_train[val_idx]
        Y_train, Y_val = Y_train_copy[train_idx], Y_train_copy[val_idx]
        # 创建数据加载器
        if type(X_train) == np.ndarray:
            X_train = torch.from_numpy(X_train).float()
        if type(X_val) == np.ndarray:
            X_val = torch.from_numpy(X_val).float()

        train_dataset = MyDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = MyDataset(X_val, Y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练循环
        best_loss = float('inf')
        best_f1 = 0
        no_improvement_count = 0

        # 创建模型实例
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
        # 计算类别权重
        num_positive = sum(Y_train)
        num_negative = len(Y_train) - num_positive

        # 计算权重
        positive_weight = num_negative / (num_positive + num_negative)
        negative_weight = num_positive / (num_positive + num_negative)

        class_weights = torch.tensor([negative_weight, positive_weight])
        # 定义损失函数并传递权重
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

        #优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                x, y = batch

                prediction = model(x)
                prediction= prediction.view(-1)

                loss = criterion(prediction, y) 
                loss.backward()     
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = torch.mean(torch.tensor(train_losses))

            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    prediction = model(x)
                    prediction = prediction.view(-1)
                    loss = criterion(prediction, y)
                    val_losses.append(loss.item())
                    val_preds.extend(torch.sigmoid(prediction).cpu().numpy().tolist())
                    val_labels.extend(y.cpu().numpy().tolist())
            avg_val_loss = torch.mean(torch.tensor(val_losses))
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            val_preds = (val_preds > 0.5).astype(int)
            val_f1 = f1_score(val_labels, val_preds)
            res_sub.append([avg_train_loss, avg_val_loss, val_f1])
            print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} Val F1: {val_f1:.4f}')

            # 判断是否早停
            if avg_val_loss < best_loss or val_f1 > best_f1:
                best_loss = avg_val_loss
                best_f1 = val_f1
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print('Early stopping!')
                    break
        res.append(res_sub)
        torch.save(model.state_dict(),config.modelname+str(fold)+".pth")
        


def __main__():
    main()

if __name__ == '__main__':
    __main__()