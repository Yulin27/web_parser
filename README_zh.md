<h1>项目概述</h1>
这个Python程序旨在使用机器学习技术(LSTM算法)通过标签路径来识别网页的正文。为此，我们将使用Python中的多个第三方库，例如BeautifulSoup、gensim 和 torch等。 

&nbsp;
&nbsp;


<h1>程序设计</h1>
在这个Python程序中，我们首先用FastText来训练一个词向量模型，以便于我们后续的机器学习模型。在训练过程中先使用BeautifulSoup库和Selenium库（待修改）解析HTML网页，然后对网页进行预处理，包括去除HTML标记、过滤噪声文本等。然后生成一个dom树并且分配对应文本。接下来，我们将使用PyTorch库来训练机器学习模型，以识别网页的正文，并用交叉验证法来评估模型的性能，并保存模型。最后，我们将使用训练好的模型来预测网页的正文。
  
&nbsp;
&nbsp;


<h1>使用方法</h1>
在使用这个Python程序之前，您需要安装所需的第三方库。您可以使用以下命令来安装它们：  

```python
pip install -r requirements.txt
```

您可以在终端中使用以下命令来预测网页正文：

```shell
python3 prediction.py -i <inputfile> -o <outputfile> -m <model>
```
- *inputfile* ：输入文件路径
- *outputfile*： 输出文件路径
- *model*：模型类型，值为 zh/la（中文模型/拉丁文模型）。

*每次运行都加载模型运行时间会较长。建议将模型缓存，以便下次运行时直接使用模型，参考exemple.ipynb*

您可以使用以下命令来训练模型：

```shell
python3 train_model.py
```
 
&nbsp;
&nbsp;

<h1>文件结构</h1>  

- dataset: 存放数据集  

- encode : 存放编码文件和编码模型  

- model : 存放训练好的模型和参数
    - model_LSTM1.pth : 中文模型
    - model_LSTM_la.pth : 拉丁文模型  
    - normalization_params_zh.pkl : 中文模型参数
    - normalization_params_la.pkl : 拉丁文模型参数
      
- label : 存放标签文件

- labeliser.py : 标签化路径

- preprocess.py : 包括预处理网页文件的函数

- encode_tag_classname_list.py : 用Fasttext编码训练集中的标签和类名

- encode_path.py : 包括编码路径相关的函数，**如要添加新的特征，在*extract_features*和*encode_feature*函数中添加即可**。

- train_model.py : 训练模型

- prediction.py : 预测网页正文
