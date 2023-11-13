import re
from bs4 import BeautifulSoup, Comment
import numpy as np
import html
from selenium import webdriver
import base64
import config

def preprocess_html(html_content):
    # 删除所有换行符
    html_content = html_content.replace('\n', '')
    html_text = html.unescape(html_content)

    # 删除所有的<!DOCTYPE>声明
    html_text = re.sub(r'<\s*!\s*?DOCTYPE.*?>', '', html_text, flags=re.DOTALL)

    soup = BeautifulSoup(html_text, 'html.parser')

    # # Remove each !DOCTYPE tag
    # for tag in doctype_tags:
    #     tag.extract()
        
    # 删除注释
    comments = soup.find_all(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    # 定义需要删除的标签
    excluded_tags = ["meta", "script", "iframe", "img","link","style", "source","br","hr"]
    html_body = soup.html
    # 删除指定的标签
    for tag in excluded_tags:
        for elem in soup(tag):
            elem.decompose()
    # 替换转义字符\
    for elem in soup():
        if elem.string:
            elem.string.replace_with(elem.string.replace('\\', ''))
    # 提取HTML的主体内容
    html_body = soup.html
    if len(soup.text)<100:
        return 'None'
    # 将主体内容转换为字符串
    html_text = str(html_body)

    return html_text


# 使用selenium模拟浏览器打开网页，以解决beautifulsoup无法解析动态网页的问题
def preprocess_html_2(html_content, filename):
    main_content = preprocess_html(html_content)
    if main_content == 'None':
        website = base64.b64decode(filename)
        website = website.decode('utf-8')

        # 根据自身的浏览器进行设置
        if config.webdriver == 'Chrome':
            options = webdriver.ChromeOptions()
            driver = webdriver.Chrome(executable_path= config.web_executable_path, options=options)
        elif config.webdriver == 'Firefox':
            options = webdriver.FirefoxOptions()
            driver = webdriver.Firefox(executable_path= config.web_executable_path, options=options)
        elif config.webdriver == 'Edge':
            options = webdriver.EdgeOptions()
            driver = webdriver.Edge(executable_path= config.web_executable_path, options=options)
        elif config.webdriver == 'Safari':
            options = webdriver.SafariOptions()
            driver = webdriver.Safari(executable_path= config.web_executable_path, options=options)
        else:
            print("Web Driver Error!")
            return None
        
        options.headless = True
        driver.get(website)
        main_content = driver.page_source
        driver.quit()
        return preprocess_html(main_content)
    
    else:
        return main_content

def dom_tree(html):
    stack = []
    # 提取标签
    tags = re.findall(r'<[^>]+>', html)
    for i in range(len(tags)):
        tag = tags[i]
        if tag.startswith('<\\/'):
            tags[i] = tag[0]+tag[3:]

    tag_paths = []
    for i in range(len(tags)):
        tag = tags[i]
        if tag.startswith('</'):  # 处理结束标签
            tag_name = tag[2:-1]  # 提取标签名
            while stack and stack[-1] != tag_name:  # 寻找对应的开始标签
                stack.pop()
            if stack:
                stack.pop()  # 从堆栈中弹出开始标签
            tag_path = stack.copy()  # 创建标签路径的副本
            tag_path.append(tag[1:-1])
            tag_paths.append(tag_path)
            # print("Tag:", tag)
            # print("End Tag Path:", tag_path)
        else:  # 处理开始标签
            tag_name = tag.split()[0][1:]
            classname = re.findall(r'class=".*?"', tag)
            if tag_name[-1]=='>':
                tag_name = tag_name[:-1] 
            stack.append(tag_name)  # 将开始标签推入堆栈
            tag_path = stack.copy()  # 创建标签路径的副本
            tag_name_complet = re.findall(r'<[^>]+>', tag)[0]
            tag_path[-1] = tag_name_complet[1:-1]
            tag_paths.append(tag_path)
            # print("Tag:", tag)
            # print("Start Tag Path:", tag_path)
    return tag_paths

def clean_content(content_list):
    for i in range(len(content_list)):
        content_list[i] = re.sub(r'\s+', ' ', content_list[i])
        # 删除特殊字符
        content_list[i] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', content_list[i])
    content_list = [x for x in content_list if (re.search(r'\S', x))]
    return content_list

def trait_html(html_content):
    # 将html内容按照标签分割
    html_content = str(html_content)
    html_content = re.split(r'((?:<[^<>]*>)?)', html_content)

    # 删除空字符
    html_content = clean_content(html_content)
    return html_content

def assign_content_to_tags(dom_tree, html_content):
    result = []
    html_content = trait_html(html_content)
    
    index = 0
    for i in range(len(dom_tree)):
        tag = dom_tree[i][-1]
        if tag.startswith('/'):
            res = ''
            while html_content[index][1:-1]!=tag and index < len(html_content):
                res+=html_content[index]
                index+=1
            index += 1

        else:
            # print(html_content[index][1:-1],tag)
            res = ''
            
            # 上一个结束标签和下一个开始标签的中间有内容的情况，此处把它分配给下一个开始标签
            if html_content[index]!='':
                test = html_content[index].split()[0][1:]
            while index < len(html_content) and (html_content[index]=='' or html_content[index][0]!='<' or html_content[index][1:-1].split()[0] != tag.split()[0]):
                res+=html_content[index]
                index += 1

            if index >= len(html_content):
                break
            if tag == html_content[index][1:-1]:
                index += 1
                while  (html_content[index]=='' or html_content[index][0] != '<') and index < len(html_content):
                    res+=html_content[index]
                    index += 1
        result.append(res)
    return result


def extract_nonempty_content_index(content):
    nonempty_content_index = []
    for i in range(len(content)):
        if re.search(r'\S', content[i]):
            nonempty_content_index.append(i)
    return nonempty_content_index

def remove_empty_content(dom_tree , content):
    dom_tree = np.asarray(dom_tree)
    content = np.asarray(content)
    nonempty_content_index = extract_nonempty_content_index(content)
    return  dom_tree[nonempty_content_index], content[nonempty_content_index]
