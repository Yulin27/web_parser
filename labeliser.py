
import os
from preprocess import *
import pandas as pd
import html
import config

# generate target label for each tag
def generate_target(content, text):

    text = re.sub(r'\s', '', text)  
    text =  re.sub(r'[^\w\s]', '', text)
    text = re.escape(text)
    text = html.unescape(text)
    target = []
    for c in content:

        if len(c) > 3:
            c = html.unescape(c)

            c = re.sub(r'\s', '', c)  
            c =  re.sub(r'[^\w\s]', '', c)
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


    df = pd.DataFrame(columns=["Filename", "Body text", "Website"])

    with open(train_text, 'r', encoding='utf-8') as f:
        lines_train = f.readlines()

    with open(test_text, 'r', encoding='utf-8') as f:
        lines_test = f.readlines()

    lines = lines_train + lines_test

    for line in lines:
        if line.startswith("{"):
            row = eval(line)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)


    for filename in os.listdir(train_html):
        if filename.endswith(".html"):

            html_file_path = os.path.join(train_html, filename)
            filename_pur = os.path.splitext(filename)[0]
            txt_file_name = filename_pur + ".txt"
            txt_file_path = os.path.join(target_folder, txt_file_name)
            with open(html_file_path, "r", encoding="utf-8") as html_file:
                    
                html_content = html_file.read()
                
                html_content = preprocess_html_2(html_content, filename_pur)
                paths = dom_tree(html_content)
                content = assign_content_to_tags(paths, html_content)
                content = clean_content(content)


                text = df[df['Filename'] == os.path.splitext(filename)[0]]['Body text'].values[0]
                target = generate_target(content, text)

                target_content = [(x, y) for x, y in zip(content, target)]

                with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                    df_target = pd.DataFrame(target_content, columns=['text', 'label']) 
                    df_target = df_target[df_target['text'].str.strip() != '']

                    df_target.reset_index(drop=False, inplace=True)

                    df_target.to_csv(target_folder+txt_file_name, sep='\t', index=False) 


if __name__ == "__main__":
    main()