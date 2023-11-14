import re
from bs4 import BeautifulSoup, Comment
import numpy as np
import html
from selenium import webdriver
import base64
import config

def preprocess_html(html_content):
    # clean html
    html_content = html_content.replace('\n', '')
    html_text = html.unescape(html_content)

    html_text = re.sub(r'<\s*!\s*?DOCTYPE.*?>', '', html_text, flags=re.DOTALL)

    soup = BeautifulSoup(html_text, 'html.parser')

    # # Remove each !DOCTYPE tag
    # for tag in doctype_tags:
    #     tag.extract()
        
    comments = soup.find_all(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    excluded_tags = ["meta", "script", "iframe", "img","link","style", "source","br","hr"]
    html_body = soup.html
    for tag in excluded_tags:
        for elem in soup(tag):
            elem.decompose()
    for elem in soup():
        if elem.string:
            elem.string.replace_with(elem.string.replace('\\', ''))

    html_body = soup.html
    if len(soup.text)<100:
        return 'None'
    html_text = str(html_body)

    return html_text


# open the dynamic web page by selenium
def preprocess_html_2(html_content, filename):
    main_content = preprocess_html(html_content)
    if main_content == 'None':
        website = base64.b64decode(filename)
        website = website.decode('utf-8')

        # set the webdriver
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
    # extract tags
    tags = re.findall(r'<[^>]+>', html)
    for i in range(len(tags)):
        tag = tags[i]
        if tag.startswith('<\\/'):
            tags[i] = tag[0]+tag[3:]

    tag_paths = []
    for i in range(len(tags)):
        tag = tags[i]
        if tag.startswith('</'):  # end tag
            tag_name = tag[2:-1]  # get tag name
            while stack and stack[-1] != tag_name:  # search for the matching start tag
                stack.pop()
            if stack:
                stack.pop()  # remove the matching start tag
            tag_path = stack.copy()  
            tag_path.append(tag[1:-1])
            tag_paths.append(tag_path)
            # print("Tag:", tag)
            # print("End Tag Path:", tag_path)
        else:  # start tag
            tag_name = tag.split()[0][1:]
            classname = re.findall(r'class=".*?"', tag)
            if tag_name[-1]=='>':
                tag_name = tag_name[:-1] 
            stack.append(tag_name)  # push tag name
            tag_path = stack.copy()  
            tag_name_complet = re.findall(r'<[^>]+>', tag)[0]
            tag_path[-1] = tag_name_complet[1:-1]
            tag_paths.append(tag_path)
            # print("Tag:", tag)
            # print("Start Tag Path:", tag_path)
    return tag_paths

def clean_content(content_list):
    for i in range(len(content_list)):
        content_list[i] = re.sub(r'\s+', ' ', content_list[i])
        # delete the invisible characters
        content_list[i] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', content_list[i])
    content_list = [x for x in content_list if (re.search(r'\S', x))]
    return content_list

def trait_html(html_content):
    # split the html content
    html_content = str(html_content)
    html_content = re.split(r'((?:<[^<>]*>)?)', html_content)

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
            
            # if there is content between a end tag and a starttag, assign it to the start tag
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
