
import os
from preprocess import *
import pandas as pd
import html
import config

# 生成目标标签列表
def generate_target(content, text):
    text = re.sub(r'\s', '', text)  # 移除所有空白字符
    text =  re.sub(r'[^\w\s]', '', text)
    # 对含有转义字符的文本进行转义处理
    text = re.escape(text)
    text = html.unescape(text)
    target = []
    for c in content:

        if len(c) > 3:
            c = html.unescape(c)

            c = re.sub(r'\s', '', c)  # 移除所有空白字符
            c =  re.sub(r'[^\w\s]', '', c)
            # 对含有转义字符的文本进行转义处理
            c = re.escape(c)
            if c in text:
                target.append(1)
            else:
                target.append(0)
        else:
            target.append(0)
    return target


def main():

    target_folder = config.target_folder
    train_text = config.train_text
    test_text = config.test_text
    train_html = config.train_html

    # 创建空的 DataFrame
    df = pd.DataFrame(columns=["Filename", "Body text", "Website"])

    with open(train_text, 'r', encoding='utf-8') as f:
        lines_train = f.readlines()

    with open(test_text, 'r', encoding='utf-8') as f:
        lines_test = f.readlines()

    lines = lines_train + lines_test
    # 解析每一行文本并添加到 DataFrame 中
    for line in lines:
        if line.startswith("{"):
            row = eval(line)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)


    for filename in os.listdir(train_html):
        if filename.endswith(".html"):
            # 构造html文件的完整路径
            html_file_path = os.path.join(train_html, filename)
            # 构造对应的txt文件名，保留原文件名，但将扩展名改为txt
            filename_pur = os.path.splitext(filename)[0]
            txt_file_name = filename_pur + ".txt"
            # 构造txt文件的完整路径
            txt_file_path = os.path.join(target_folder, txt_file_name)
            # 读取html文件内容
            with open(html_file_path, "r", encoding="utf-8") as html_file:
                    
                html_content = html_file.read()
                # 进行处理，将content内容写入txt文件
                html_content = preprocess_html_2(html_content, filename_pur)
                paths = dom_tree(html_content)
                content = assign_content_to_tags(paths, html_content)
                content = clean_content(content)


                # 调用函数生成目标标签
                text = df[df['Filename'] == os.path.splitext(filename)[0]]['Body text'].values[0]
                target = generate_target(content, text)

                # 使用zip函数将两个列表合并
                target_content = [(x, y) for x, y in zip(content, target)]

                with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                    # 将表格数据转换为DataFrame
                    df_target = pd.DataFrame(target_content, columns=['text', 'label']) 
                    # 删除文本为空字符串的行
                    df_target = df_target[df_target['text'].str.strip() != '']

                    # 重新设置索引，并保持原来的索引不变
                    df_target.reset_index(drop=False, inplace=True)

                    # 将DataFrame保存为txt文件
                    df_target.to_csv(target_folder+txt_file_name, sep='\t', index=False) 


if __name__ == "__main__":
    main()