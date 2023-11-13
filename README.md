<h1>Project Overview</h1>
This Python program aims to identify the main body of a web page using machine learning techniques (LSTM algorithm) through labeled paths. For this, we will use multiple third-party libraries in Python, such as BeautifulSoup, gensim, and torch, etc.

&nbsp;
&nbsp;

<h1>Program Design</h1>
In this Python program, we first use FastText to train a word vector model, which will facilitate our subsequent machine learning model. During the training process, we use the BeautifulSoup and Selenium libraries (to be modified) to parse HTML web pages, then preprocess the web pages, including removing HTML tags, filtering out noise text, etc. Then, we generate a DOM tree and assign corresponding text. Next, we will use the PyTorch library to train the machine learning model to identify the main body of the web page, evaluate the performance of the model with cross-validation, and save the model. Finally, we will use the trained model to predict the main body of web pages.

&nbsp;
&nbsp;

<h1>Usage</h1>
Before using this Python program, you need to install the required third-party libraries. You can use the following command to install them:

```python
pip install -r requirements.txt
```



You can use the following command to train the model:

```shell
python3 train_model.py
```

You can use the following command in the terminal to predict the main body of web pages:

```shell
python3 prediction.py -i <inputfile> -o <outputfile> -m <model>
```
- *inputfile*: Input file path
- *outputfile*: Output file path
- *model*: Model type


&nbsp;
&nbsp;

<h1>File Structure</h1>

- dataset: Holds the dataset

- encode: Holds encoding files and encoding models

- model: Holds trained models and parameters
    
- label: Holds label files

- labeliser.py: Path labeling

- preprocess.py: Includes functions for preprocessing web page files

- encode_tag_classname_list.py: Use Fasttext to encode tags and class names in the training set

- encode_path.py: Includes functions related to encoding paths, **to add new features, add them in the *extract_features* and *encode_feature* functions**.

- train_model.py: Train the model

- prediction.py: Predict the main body of web pages
