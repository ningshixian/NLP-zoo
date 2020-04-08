# NLP-zoo

> 对中文的NLP资源做个汇总，以备不时之需

# 目录

* [中文NLP语料资源](#中文NLP语料资源)
* [NLP工具包](#NLP工具包)
* [中文分词](#中文分词)
* [QA & Chatbot 问答和聊天机器人](#QA & Chatbot 问答和聊天机器人)

* [NLP相关论文](#NLP相关论文)
* [Python-Tutorial](#Python-Tutorial)
* [matplotlib-Ipython](#matplotlib-Ipython)
* [pandas](#pandas)
* [scikit-learn](#scikit-learn)


# 中文NLP语料资源

- **闲聊常用语料**

| 语料名称 | 语料Size | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [中文对白语料 chinese conversation corpus](https://github.com/fate233/dgk_lost_conv) |  |  | 可以用作聊天机器人的训练语料 |
| [chatterbot](https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese) | 560 | 开源项目 | 按类型分类，质量较高 |
| qingyun（青云语料） | 10W | 某聊天机器人交流群 | 相对不错，生活化 |
| [xiaohuangji（小黄鸡语料）](https://github.com/candlewill/Dialog_Corpus) | 45W | 	原人人网项目语料 | 有一些不雅对话，少量噪音 |
| [douban（豆瓣多轮）](https://github.com/MarkWuNLP/MultiTurnResponseSelection) | 352W | 来自北航和微软的paper, 开源项目 | 噪音相对较少，原本是多轮（平均7.6轮） |
| [weibo（微博语料）]() | 443W | 来自华为的paper | 有一些噪音 |

```
使用方法
下载语料 https://pan.baidu.com/s/1szmNZQrwh9y994uO8DFL_A 提取码：f2ex

执行命令即可
python main.py

生成结果
每个来源的语料分别生成一个独立的*.tsv文件，都放在新生成的clean_chat_corpus文件夹下。
生成结果格式为 tsv格式，每行是一个样本，先是query，再是answer
```

- **领域特定语料**

| 语料名称 | 语料大小 | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [保险行业QA语料库](https://github.com/Samurais/insuranceqa-corpus-zh)  | 未知 | 通过翻译 insuranceQA产生的数据集 | train_data含有问题12,889条，数据 141779条，正例：负例 = 1:10； test_data含有问题2,000条，数据 22000条，正例：负例 = 1:10；valid_data含有问题2,000条，数据 22000条，正例：负例 = 1:10 |
| [翻译语料(translation2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/translation2019zh.zip) | 520万个中英文平行语料( 原始数据1.1G，压缩文件596M) | 单元格 | 中英文平行语料520万对 |


- **命名实体识别NER语料**

| 语料名称 | 语料大小 | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [(中文) weibo NER corpus](https://github.com/hltcoe/golden-horse) | 未知 | 未知 | 包含了1,890条，设计的实体有：人名、地点、组织、地理政治相关实体 |
| [boson数据](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/boson) | 未知 | 未知 | 包含6种实体类型 |
| [人民日报数据集](https://pan.baidu.com/s/1LDwQjoj7qc-HT9qwhJ3rcA) | 未知 | 未知 | password: 1fa3 |
| [MSRA微软亚洲研究院数据集](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA) | 未知 | 未知 | 5 万多条中文命名实体识别标注数据（IOB2 格式，符合 CoNLL 2002 和 CRF++ 标准） |
| [Resume NER data](https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER) |  | ACL 2018 paper 《Chinese NER Using Lattice LSTM》 | 爬虫新浪财经的的简历数据, CoNLL format (BIOES tag scheme)，包括城市、学校、地点、人名、组织等 |


- **情感/观点/评论 倾向性分析**

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| ChnSentiCorp_htl_all | 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论 | [点击查看](./datasets/ChnSentiCorp_htl_all/intro.ipynb) |
| waimai_10k | 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条 | [点击查看](./datasets/waimai_10k/intro.ipynb) |
| online_shopping_10_cats | 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，<br /> 包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店 | [点击查看](./datasets/online_shopping_10_cats/intro.ipynb) |
| weibo_senti_100k | 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条 | [点击查看](./datasets/weibo_senti_100k/intro.ipynb) |
| simplifyweibo_4_moods | 36 万多条，带情感标注 新浪微博，包含 4 种情感，<br /> 其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条 | [点击查看](./datasets/simplifyweibo_4_moods/intro.ipynb) |
| dmsc_v2 | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 | [点击查看](./datasets/dmsc_v2/intro.ipynb) |
| yf_dianping | 24 万家餐馆，54 万用户，440 万条评论/评分数据 | [点击查看](./datasets/yf_dianping/intro.ipynb) |
| yf_amazon | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb) |


- **推荐系统**

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| ez_douban | 5 万多部电影（3 万多有电影名称，2 万多没有电影名称），2.8 万 用户，280 万条评分数据 | [点击查看](./datasets/ez_douban/intro.ipynb) |
| dmsc_v2 | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 | [点击查看](./datasets/dmsc_v2/intro.ipynb) |
| yf_dianping | 24 万家餐馆，54 万用户，440 万条评论/评分数据 | [点击查看](./datasets/yf_dianping/intro.ipynb) |
| yf_amazon | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb) |


- **FAQ 问答系统**

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| 保险知道 | 8000 多条保险行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/baoxianzhidao/intro.ipynb) |
| 安徽电信知道 | 15.6 万条电信问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/anhuidianxinzhidao/intro.ipynb) |
| 金融知道 | 77 万条金融行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/financezhidao/intro.ipynb) |
| 法律知道 | 3.6 万条法律问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/lawzhidao/intro.ipynb) |
| 联通知道 | 20.3 万条联通问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/liantongzhidao/intro.ipynb) |
| 农行知道 | 4 万条农业银行问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/nonghangzhidao/intro.ipynb) |
| 保险知道 | 58.8 万条保险行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/baoxianzhidao/intro.ipynb) |


- **超大型通用语料**

| 语料名称 | 语料大小 | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [维基百科json版(wiki2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/wiki_zh_2019.zip) | 104万个词条, 1.6G | wiki | 做预训练的语料或构建词向量，也可以用于构建知识问答 |
| [新闻语料json版(news2016zh)](https://pan.baidu.com/s/1LJeq1dkA0wmYd9ZGZw72Xg) | 250万篇新闻,原始数据9G | 涵盖了6.3万个媒体，含标题、关键词、描述、正文 | 密码: film 包含了250万篇新闻。数据集划分：数据去重并分成三个部分。训练集：243万；验证集：7.7万；测试集，数万 |
| [百科类问答json版(baike2018qa)](https://pan.baidu.com/s/12TCEwC_Q3He65HtPKN17cA) | 150万个问答,原始数据1G多 | 密码:fu45 | 含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别 |
| [社区问答json版(webtext2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/webtext2019zh.zip) | 410万个问答,过滤后数据3.7G | 1400万个原始问答 | 含有410万个预先过滤过的、高质量问题和回复。 |
| 单元格 | 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 | 单元格 |

- 
- **[中文同义词表，反义词表，否定词表](https://github.com/guotong1988/chinese_dictionary)**

- **腾讯词向量**

> 腾讯AI实验室公开的中文词向量数据集包含800多万中文词汇，其中每个词对应一个200维的向量。

> 下载地址：https://ai.tencent.com/ailab/nlp/embedding.html


# NLP工具包

- [THULAC 中文词法分析工具包](http://thulac.thunlp.org/) by 清华 (C++/Java/Python)

- [LTP 语言技术平台](https://github.com/HIT-SCIR/ltp) by 哈工大 (C++)  [pylyp](https://github.com/HIT-SCIR/pyltp) LTP的python封装

- [BaiduLac](https://github.com/baidu/lac) by 百度 Baidu's open-source lexical analysis tool for Chinese, including word segmentation, part-of-speech tagging & named entity recognition. 

- [HanLP](https://github.com/hankcs/HanLP) (Java)

- [SnowNLP](https://github.com/isnowfy/snownlp) (Python) Python library for processing Chinese text

- [小明NLP](https://github.com/SeanLee97/xmnlp) (Python) 轻量级中文自然语言处理工具

- [chinese_nlp](https://github.com/taozhijiang/chinese_nlp) (C++ & Python) Chinese Natural Language Processing tools and examples

- [CoreNLP](https://github.com/stanfordnlp/CoreNLP) by Stanford (Java) A Java suite of core NLP tools.

- [Stanza](https://github.com/stanfordnlp/stanza) by Stanford (Python) A Python NLP Library for Many Human Languages

- [spaCy](https://spacy.io/) (Python) Industrial-Strength Natural Language Processing with a [online course](https://course.spacy.io/)

- [gensim](https://github.com/RaRe-Technologies/gensim) (Python) Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. 

- [Kashgari](https://github.com/BrikerMan/Kashgari) - Simple and powerful NLP framework, build your state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS) and text classification tasks. Includes BERT and word2vec embedding.

  
# 中文分词

- [Jieba 结巴中文分词](https://github.com/fxsjy/jieba) (Python及大量其它编程语言衍生) 做最好的 Python 中文分词组件

- [北大中文分词工具](https://github.com/lancopku/pkuseg-python) (Python) 高准确度中文分词工具，简单易用，跟现有开源工具相比大幅提高了分词的准确率。

- [A neural network model for Chinese named entity recognition](https://github.com/zjy-ucas/ChineseNER)

- [bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner) 使用预训练语言模型BERT做中文NER
  
- [Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese) Chinese Named Entity Recognition with IDCNN/biLSTM+CRF, and Relation Extraction with biGRU+2ATT 中文实体识别与关系提取

  
# QA & Chatbot 问答和聊天机器人 

- [Rasa NLU](https://github.com/RasaHQ/rasa_nlu) (Python) turn natural language into structured data, a Chinese fork at [Rasa NLU Chi](https://github.com/crownpku/Rasa_NLU_Chi)

- [Rasa Core](https://github.com/RasaHQ/rasa_core) (Python) machine learning based dialogue engine for conversational software

- [Chatstack](https://github.com/crownpku/Chatstack-Doc) A Full Pipeline UI for building Chinese NLU System

- [Snips NLU](https://github.com/snipsco/snips-nlu) (Python) Snips NLU is a Python library that allows to parse sentences written in natural language and extracts structured information.

- [DeepPavlov](https://github.com/deepmipt/DeepPavlov) (Python) An open source library for building end-to-end dialog systems and training chatbots.

- [ChatScript](https://github.com/bwilcox-1234/ChatScript) Natural Language tool/dialog manager, a rule-based chatbot engine.

- [Chatterbot](https://github.com/gunthercox/ChatterBot) (Python) ChatterBot is a machine learning, conversational dialog engine for creating chat bots.

- [Chatbot](https://github.com/zake7749/Chatbot) (Python) 基於向量匹配的情境式聊天機器人

- [QA-Snake](https://github.com/SnakeHacker/QA-Snake) (Python) 基于多搜索引擎和深度学习技术的自动问答

- [使用TensorFlow实现的Sequence to Sequence的聊天机器人模型](https://github.com/qhduan/Seq2Seq_Chatbot_QA) (Python)

- [使用深度学习算法实现的中文阅读理解问答系统](https://github.com/S-H-Y-GitHub/QA) (Python)

- [AnyQ by Baidu](https://github.com/baidu/AnyQ) 主要包含面向FAQ集合的问答系统框架、文本语义匹配工具SimNet。

- [QASystemOnMedicalKG](https://github.com/liuhuanyong/QASystemOnMedicalKG) (Python) 以疾病为中心的一定规模医药领域知识图谱，并以该知识图谱完成自动问答与分析服务。

- [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat) (Python) 用于中文闲聊的GPT2模型


# NLP相关论文

### Attention
1. [SEO, Minjoon, et al. Bidirectional attention flow for machine comprehension. arXiv preprint arXiv:1611.01603, 2016.](https://arxiv.org/pdf/1611.01603)

2. [ZADEH, Amir, et al. Multi-attention recurrent network for human communication comprehension. arXiv preprint arXiv:1802.00923, 2018.](https://arxiv.org/pdf/1802.00923)

3. [CHEN, Kehai, et al. Syntax-Directed Attention for Neural Machine Translation. arXiv preprint arXiv:1711.04231, 2017.](https://arxiv.org/pdf/1711.04231)

### 词向量表示学习

1. [AutoExtend_ACL2014](https://arxiv.org/pdf/1507.01127)

2. [AutoExtend_ACL2017](https://www.mitpressjournals.org/doi/full/10.1162/COLI_a_00294)

3. [WELLER-DI MARCO, Marion; FRASER, Alexander; IM WALDE, Sabine Schulte. Addressing Problems across Linguistic Levels in SMT: Combining Approaches to Model Morphology, Syntax and Lexical Choice. In: Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. 2017. p. 625-630.](http://www.aclweb.org/anthology/E17-2099)

4. [YAGHOOBZADEH, Yadollah; SCHÜTZE, Hinrich. Multi-level representations for fine-grained typing of knowledge base entities. arXiv preprint arXiv:1701.02025, 2017.](https://arxiv.org/pdf/1701.02025)

5. [TISSIER, Julien; GRAVIER, Christophe; HABRARD, Amaury. Dict2vec: Learning Word Embeddings using Lexical Dictionaries. In: Conference on Empirical Methods in Natural Language Processing (EMNLP 2017). 2017. p. 254-263.](https://hal-ujm.archives-ouvertes.fr/ujm-01613953/file/emnlp2017.pdf)

6. [PINTER, Yuval; GUTHRIE, Robert; EISENSTEIN, Jacob. Mimicking word embeddings using subword RNNs. arXiv preprint arXiv:1707.06961, 2017.](https://arxiv.org/pdf/1707.06961)

7. [CHIU, Billy, et al. How to train good word embeddings for biomedical NLP. In: Proceedings of the 15th Workshop on Biomedical Natural Language Processing. 2016. p. 166-174.](http://www.aclweb.org/anthology/W16-2922)

8. [XIE, Ruobing, et al. Lexical sememe prediction via word embeddings and matrix factorization. In: Proceedings of the 26th International Joint Conference on Artificial Intelligence. AAAI Press, 2017. p. 4200-4206.](https://www.ijcai.org/proceedings/2017/0587.pdf)

9. [CHE, Zhengping, et al. Exploiting convolutional neural network for risk prediction with medical feature embedding. arXiv preprint arXiv:1701.07474, 2017.](https://arxiv.org/pdf/1701.07474)

10. [YU, Liang-Chih, et al. Refining word embeddings for sentiment analysis. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 534-539.](http://www.aclweb.org/anthology/D17-1056)

11. [FARUQUI, Manaal, et al. Retrofitting word vectors to semantic lexicons. arXiv preprint arXiv:1411.4166, 2014.](https://arxiv.org/pdf/1411.4166)

12. [ABEND, Omri; RAPPOPORT, Ari. The state of the art in semantic representation. In: Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017. p. 77-89.](http://www.aclweb.org/anthology/P17-1008)

13. [WU, Ledell, et al. StarSpace: Embed All The Things!. arXiv preprint arXiv:1709.03856, 2017.](https://arxiv.org/pdf/1709.03856)

14. [CHEN, Liang-Wei; LEE, Wei-Chun; HWANG, Hsiang-Wei. When Word Embedding Meets Lexical Networks.](https://pdfs.semanticscholar.org/52f9/c705b4303576108cab5a6b22f73f0e7d29af.pdf)


### 对抗GAN

1. [王坤峰, et al. 生成式对抗网络 GAN 的研究进展与展望. 自动化学报, 2017, 43.3: 321-332.](http://html.rhhz.net/ZDHXBZWB/html/20170301.htm)

2. [CHEN, Xinchi, et al. Adversarial multi-criteria learning for chinese word segmentation. arXiv preprint arXiv:1704.07556, 2017.](https://arxiv.org/pdf/1704.07556)

3. [LIU, Pengfei; QIU, Xipeng; HUANG, Xuanjing. Adversarial multi-task learning for text classification. arXiv preprint arXiv:1704.05742, 2017.](https://arxiv.org/pdf/1704.05742)

4. [LI, Zheng, et al. End-to-end adversarial memory network for cross-domain sentiment classification. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2017. p. 2237.](https://www.ijcai.org/proceedings/2017/0311.https://arxiv.org/pdf/1710.07035pdf)

5. [GUI, Tao, et al. Part-of-speech tagging for twitter with adversarial neural networks. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2411-2420.](http://www.aclweb.org/anthology/D17-1256)

6. [KIM, Joo-Kyung, et al. Cross-Lingual Transfer Learning for POS Tagging without Cross-Lingual Resources. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2832-2838.](http://www.aclweb.org/anthology/D17-1302)

7. [CRESWELL, Antonia, et al. Generative Adversarial Networks: An Overview. IEEE Signal Processing Magazine, 2018, 35.1: 53-65.](https://arxiv.org/pdf/1710.07035)

### 多任务学习

1. [CRICHTON, Gamal, et al. A neural network multi-task learning approach to biomedical named entity recognition. BMC bioinformatics, 2017, 18.1: 368.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8)

2. [Chen, X., Qiu, X., & Huang, X. (2016). A feature-enriched neural model for joint Chinese word segmentation and part-of-speech tagging. arXiv preprint arXiv:1611.05384.](https://arxiv.org/pdf/1611.05384)

3. [RUDER, Sebastian. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.](https://arxiv.org/pdf/1706.05098)

4. [LONG, Mingsheng, et al. Learning Multiple Tasks with Multilinear Relationship Networks. In: Advances in Neural Information Processing Systems. 2017. p. 1593-1602.](http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-multilinear-relationship-networks.pdf)

5. [AGUILAR, Gustavo, et al. A Multi-task Approach for Named Entity Recognition in Social Media Data. In: Proceedings of the 3rd Workshop on Noisy User-generated Text. 2017. p. 148-153.](http://www.aclweb.org/anthology/W17-4419)


### 关系抽取任务

1. [WU, Yi; BAMMAN, David; RUSSELL, Stuart. Adversarial training for relation extraction. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 1778-1783.](http://www.aclweb.org/anthology/D17-1187)

2. [HUANG, Yi Yao; WANG, William Yang. Deep Residual Learning for Weakly-Supervised Relation Extraction. arXiv preprint arXiv:1707.08866, 2017.](https://arxiv.org/pdf/1707.08866)

3. [HUANG, Yi Yao; WANG, William Yang. Deep Residual Learning for Weakly-Supervised Relation Extraction. arXiv preprint arXiv:1707.08866, 2017.](http://www.aclweb.org/anthology/D17-1182)

4. [HE, Zhengqiu, et al. SEE: Syntax-aware Entity Embedding for Neural Relation Extraction. arXiv preprint arXiv:1801.03603, 2018.](https://arxiv.org/pdf/1801.03603)

5. [GANEA, Octavian-Eugen; HOFMANN, Thomas. Deep Joint Entity Disambiguation with Local Neural Attention. arXiv preprint arXiv:1704.04920, 2017.](https://arxiv.org/pdf/1704.04920)

6. [ADEL, Heike; SCHÜTZE, Hinrich. Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification. arXiv preprint arXiv:1707.07719, 2017.](https://arxiv.org/pdf/1707.07719)

7. [Zeng, W., Lin, Y., Liu, Z., & Sun, M. (2016). Incorporating relation paths in neural relation extraction. arXiv preprint arXiv:1609.07479.](https://arxiv.org/pdf/1609.07479)

8. [TAY, Yi; LUU, Anh Tuan; HUI, Siu Cheung. Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis. arXiv preprint arXiv:1712.05403, 2017.](https://arxiv.org/pdf/1712.05403)

9. [Zeng, X., He, S., Liu, K., & Zhao, J. (2018). Large Scaled Relation Extraction with Reinforcement Learning. Relation, 2, 3.](http://159.226.21.68/bitstream/173211/20626/1/Large%20Scaled%20Relation%20Extraction%20with%20Reinforcement%20Learning.pdf)


### 迁移学习

1. [KIM, Joo-Kyung, et al. Cross-Lingual Transfer Learning for POS Tagging without Cross-Lingual Resources. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2832-2838.](http://www.aclweb.org/anthology/D17-1302)

2. [YANG, Zhilin; SALAKHUTDINOV, Ruslan; COHEN, William W. Transfer learning for sequence tagging with hierarchical recurrent networks. arXiv preprint arXiv:1703.06345, 2017.](https://arxiv.org/pdf/1703.06345)

3. [PAN, Sinno Jialin; YANG, Qiang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 2010, 22.10: 1345-1359.](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)

4. [PAN, Sinno Jialin, et al. Domain adaptation via transfer component analysis. IEEE Transactions on Neural Networks, 2011, 22.2: 199-210.](http://www.aaai.org/ocs/index.php/IJCAI/IJCAI-09/paper/download/294/962)


### 情感分类 

1. [WANG, Bailin; LU, Wei. Learning Latent Opinions for Aspect-level Sentiment Classification. 2018.](http://www.statnlp.org/wp-content/uploads/papers/2018/Learning-Latent/absa.pdf)


### 生物医学实体识别

## Survey Papers ##
1. **Overview of BioCreative II gene mention recognition**. Smith L, Tanabe L K, nee Ando R J, et al. *Genome biology*, 2008, 9(2): S2. [[paper]](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2008-9-s2-s2)
2. **Biomedical named entity recognition: a survey of machine-learning tools**. Campos D, Matos S, Oliveira J L. *Theory and Applications for Advanced Text Mining*, 2012: 175-195. [[paper]](https://books.google.com.hk/books?hl=zh-CN&lr=&id=EfqdDwAAQBAJ&oi=fnd&pg=PA175&ots=WEKIblRekC&sig=FWoufJtWVSDHD3gbWaZXruEOiEs&redir_esc=y#v=onepage&q&f=false)
2. **Chemical named entities recognition: a review on approaches and applications**.  Eltyeb S, Salim N. *Journal of cheminformatics*, 2014, 6(1): 17. [[paper]](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-17)
3. **CHEMDNER: The drugs and chemical names extraction challenge**. Krallinger M, Leitner F, Rabal O, et al. *Journal of cheminformatics*, 2015, 7(1): S1. [[paper]](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S1)
4. **A comparative study for biomedical named entity recognition**. Wang X, Yang C, Guan R. *International Journal of Machine Learning and Cybernetics*, 2015, 9(3): 373-382. [[paper]](https://link.springer.com/article/10.1007/s13042-015-0426-6)

## Dictionary-based Methods ##
1. **Using BLAST for identifying gene and protein names in journal articles**. Krauthammer M, Rzhetsky A, Morozov P, et al. *Gene*, 2000, 259(1-2): 245-252. [[paper]](https://www.sciencedirect.com/science/article/pii/S0378111900004315)
2. **Boosting precision and recall of dictionary-based protein name recognition**. Tsuruoka Y, Tsujii J. *Proceedings of the ACL 2003 workshop on Natural language processing in biomedicine-Volume 13*, 2003: 41-48. [[paper]](https://aclanthology.info/pdf/W/W03/W03-1306.pdf)
2. **Exploiting the performance of dictionary-based bio-entity name recognition in biomedical literature**. Yang Z, Lin H, Li Y. *Computational Biology and Chemistry*, 2008, 32(4): 287-291. [[paper]](https://www.sciencedirect.com/science/article/pii/S1476927108000340)
2. **A dictionary to identify small molecules and drugs in free text**. Hettne K M, Stierum R H, Schuemie M J, et al. *Bioinformatics*, 2009, 25(22): 2983-2991. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/25/22/2983/180399) [[dictionary]](https://biosemantics.org/index.php/resources/jochem)
3. **LINNAEUS: a species name identification system for biomedical literature**. Gerner M, Nenadic G, Bergman C M. *BMC bioinformatics*, 2010, 11(1): 85. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-85)

## Rule-based Methods ##
1. **Toward information extraction: identifying protein names from biological papers**. Fukuda K, Tsunoda T, Tamura A, et al. *Pac symp biocomput*. 1998, 707(18): 707-718. [[paper]](https://pdfs.semanticscholar.org/335e/8b19ea50d3af6fcefe6f8421e2c9c8936f3f.pdf)
2. **A biological named entity recognizer**. Narayanaswamy M, Ravikumar K E, Vijay-Shanker K. *Biocomputing* 2003. 2002: 427-438. [[paper]](https://www.worldscientific.com/doi/abs/10.1142/9789812776303_0040)
3. **ProMiner: rule-based protein and gene entity recognition**. Hanisch D, Fundel K, Mevissen H T, et al. *BMC bioinformatics*, 2005, 6(1): S14. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-6-S1-S14)
4. **MutationFinder: a high-performance system for extracting point mutation mentions from text**. Caporaso J G, Baumgartner Jr W A, Randolph D A, et al. *Bioinformatics*, 2007, 23(14): 1862-1865. [[paper]](https://academic.oup.com/bioinformatics/article/23/14/1862/188647) [[code]](http://mutationfinder.sourceforge.net/)
3. **Drug name recognition and classification in biomedical texts: a case study outlining approaches underpinning automated systems**. Segura-Bedmar I, Martínez P, Segura-Bedmar M.  *Drug discovery today*, 2008, 13(17-18): 816-823. [[paper]](https://www.sciencedirect.com/science/article/pii/S1359644608002171)
3. **Investigation of unsupervised pattern learning techniques for bootstrap construction of a medical treatment lexicon**. Xu R, Morgan A, Das A K, et al. *Proceedings of the workshop on current trends in biomedical natural language processing*, 2009: 63-70. [[paper]](http://www.aclweb.org/anthology/W09-1308)
4. **Linguistic approach for identification of medication names and related information in clinical narratives**.  Hamon T, Grabar N. *Journal of the American Medical Informatics Association*, 2010, 17(5): 549-554. [[paper]](https://academic.oup.com/jamia/article/17/5/549/831598)
5. **SETH detects and normalizes genetic variants in text**. Thomas P, Rocktäschel T, Hakenberg J, et al. *Bioinformatics*, 2016, 32(18): 2883-2885. [[paper]](https://academic.oup.com/bioinformatics/article/32/18/2883/1743171) [[code]](http://rockt.github.io/SETH/)
5. **PENNER: Pattern-enhanced Nested Named Entity Recognition in Biomedical Literature**. Wang X, Zhang Y, Li Q, et al. *2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*. 2018: 540-547. [[paper]](https://ieeexplore.ieee.org/abstract/document/8621485/)


## Machine Learning-based Methods  ##

- **SVM-based Methods**

1. **Tuning support vector machines for biomedical named entity recognition**. Kazama J, Makino T, Ohta Y, et al. *Proceedings of the ACL-02 workshop on Natural language processing in the biomedical domain-Volume 3*, 2002: 1-8. [[paper]](https://aclanthology.info/pdf/W/W02/W02-0301.pdf)
2. **Biomedical named entity recognition using two-phase model based on SVMs**. Lee K J, Hwang Y S, Kim S, et al. *Journal of Biomedical Informatics*, 2004, 37(6): 436-447. [[paper]](https://www.sciencedirect.com/science/article/pii/S1532046404000863)
3. **Exploring deep knowledge resources in biomedical name recognition**. GuoDong Z, Jian S. *Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications*, 2004: 96-99. [[paper]](https://aclanthology.info/pdf/W/W04/W04-1219.pdf)

- **HMM-based Methods**

1. **Named entity recognition in biomedical texts using an HMM model**. Zhao S. *Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and its Applications*, 2004: 84-87.[[paper]](https://aclanthology.info/pdf/W/W04/W04-1216.pdf)
2. **Annotation of chemical named entities**. Corbett P, Batchelor C, Teufel S. P*roceedings of the Workshop on BioNLP 2007: Biological, Translational, and Clinical Language Processing*, 2007: 57-64. [[paper]](https://aclanthology.info/pdf/W/W07/W07-1008.pdf)
1. **Conditional random fields vs. hidden markov models in a biomedical named entity recognition task**. Ponomareva N, Rosso P, Pla F, et al. *Proc. of Int. Conf. Recent Advances in Natural Language Processing, RANLP*. 2007, 479: 483.[[paper]](http://clg.wlv.ac.uk/papers/Ponomareva-RANLP-07.pdf)

- **MEMM-based Mehtods**

1. **Cascaded classifiers for confidence-based chemical named entity recognition**. Corbett P, Copestake A. *BMC bioinformatics*, 2008, 9(11): S4. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-S11-S4)
2. **OSCAR4: a flexible architecture for chemical text-mining**. Jessop D M, Adams S E, Willighagen E L, et al. *Journal of cheminformatics*, 2011, 3(1): 41. [[paper]](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-3-41)

- **CRF-based Methods**

1. **ABNER: an open source tool for automatically tagging genes, proteins and other entity names in text**. Settles B. *Bioinformatics*, 2005, 21(14): 3191-3192.[[paper]](https://academic.oup.com/bioinformatics/article/21/14/3191/266815)
2.  **BANNER: an executable survey of advances in biomedical named entity recognition**. Leaman R, Gonzalez G. *Biocomputing* 2008. 2008: 652-663.[[paper]](https://psb.stanford.edu/psb-online/proceedings/psb08/leaman.pdf)
3.  **Detection of IUPAC and IUPAC-like chemical names**. Klinger R, Kolářik C, Fluck J, et al. *Bioinformatics*, 2008, 24(13): i268-i276. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/24/13/i268/235854)
3.  **Incorporating rich background knowledge for gene named entity classification and recognition**. Li Y, Lin H, Yang Z. *BMC bioinformatics*, 2009, 10(1): 223. [[paper]](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/1471-2105-10-223)
3.  **A study of machine-learning-based approaches to extract clinical entities and their assertions from discharge summaries**. Jiang M, Chen Y, Liu M, et al. *Journal of the American Medical Informatics Association*, 2011, 18(5): 601-606. [[paper]](https://academic.oup.com/jamia/article/18/5/601/834186)
4.   **ChemSpot: a hybrid system for chemical named entity recognition**. Rocktäschel T, Weidlich M, Leser U. *Bioinformatics*, 2012, 28(12): 1633-1640. [[paper]](https://academic.oup.com/bioinformatics/article/28/12/1633/266861)
3.  **Gimli: open source and high-performance biomedical name recognition**. Campos D, Matos S, Oliveira J L. *BMC bioinformatics*, 2013, 14(1): 54. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-54)
4.  **tmVar: a text mining approach for extracting sequence variants in biomedical literature**. Wei C H, Harris B R, Kao H Y, et al. *Bioinformatics*, 2013, 29(11): 1433-1439. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/29/11/1433/220291) [[code]](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/)
4.  **Evaluating word representation features in biomedical named entity recognition tasks**. Tang B, Cao H, Wang X, et al. *BioMed research international*, 2014, 2014. [[paper]](http://downloads.hindawi.com/journals/bmri/2014/240403.pdf)
5.  **Drug name recognition in biomedical texts: a machine-learning-based method**. He L, Yang Z, Lin H, et al. *Drug discovery today*, 2014, 19(5): 610-617. [[paper]](https://www.sciencedirect.com/science/article/pii/S1359644613003322)
3.  **tmChem: a high performance approach for chemical named entity recognition and normalization**. Leaman R, Wei C H, Lu Z. *Journal of cheminformatics*, 2015, 7(1): S3. [[paper]](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S3)
4.  **GNormPlus: an integrative approach for tagging genes, gene families, and protein domains**. Wei C H, Kao H Y, Lu Z. *BioMed research international*, 2015, 2015. [[paper]](http://downloads.hindawi.com/journals/bmri/2015/918710.pdf)
5.  **Mining chemical patents with an ensemble of open systems**[J]. Leaman R, Wei C H, Zou C, et al. *Database*, 2016, 2016. [[paper]](https://academic.oup.com/database/article-abstract/doi/10.1093/database/baw065/2630406)
6. **nala: text mining natural language mutation mentions**.  Cejuela J M, Bojchevski A, Uhlig C, et al. *Bioinformatics*, 2017, 33(12): 1852-1858. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/33/12/1852/2991428)


- **Neural Network-based Methods**

1. **Recurrent neural network models for disease name recognition using domain invariant features**. Sahu S, Anand A. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*. 2016: 2216-2225. [[paper]](https://www.aclweb.org/anthology/P16-1209)
2. **Deep learning with word embeddings improves biomedical named entity recognition**. Habibi M, Weber L, Neves M, et al. *Bioinformatics*, 2017, 33(14): i37-i48. [[paper]](https://academic.oup.com/bioinformatics/article/33/14/i37/3953940)
3.  **A neural joint model for entity and relation extraction from biomedical text**. Li F, Zhang M, Fu G, et al. *BMC bioinformatics*, 2017, 18(1): 198. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1609-9)
2. **A neural network multi-task learning approach to biomedical named entity recognition**. Crichton G, Pyysalo S, Chiu B, et al. *BMC bioinformatics*, 2017, 18(1): 368. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8) [[code]](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)
3. **Disease named entity recognition from biomedical literature using a novel convolutional neural network**. Zhao Z, Yang Z, Luo L, et al. *BMC medical genomics*, 2017, 10(5): 73. [[paper]](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-017-0316-8)
3. **An attention-based BiLSTM-CRF approach to document-level chemical named entity recognition**. Luo L, Yang Z, Yang P, et al. *Bioinformatics*, 2018, 34(8): 1381-1388. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/34/8/1381/4657076) [[code]](https://github.com/lingluodlut/Att-ChemdNER)
4. **GRAM-CNN: a deep learning approach with local context for named entity recognition in biomedical text**.  Zhu Q, Li X, Conesa A, et al. *Bioinformatics*, 2018, 34(9): 1547-1554. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/34/9/1547/4764002) [[code]](https://github.com/valdersoul/GRAM-CNN)
4. **D3NER: biomedical named entity recognition using CRF-biLSTM improved with fine-tuned embeddings of various linguistic information**. Dang T H, Le H Q, Nguyen T M, et al. *Bioinformatics*, 2018, 34(20): 3539-3546. [[paper]](https://academic.oup.com/bioinformatics/article/34/20/3539/4990492) [[code]](https://github.com/aidantee/D3NER)
4. **Transfer learning for biomedical named entity recognition with neural networks**. Giorgi J M, Bader G D. *Bioinformatics*, 2018, 34(23): 4087-4094. [[paper]](https://academic.oup.com/bioinformatics/article/34/23/4087/5026661)
5. **Label-Aware Double Transfer Learning for Cross-Specialty Medical Named Entity Recognition**. Wang Z, Qu Y, Chen L, et al. *NAACL*. 2018: 1-15. [[paper]](https://www.aclweb.org/anthology/N18-1001)
6. **Recognizing irregular entities in biomedical text via deep neural networks**. Li F, Zhang M, Tian B, et al. *Pattern Recognition Letters*, 2018, 105: 105-113. [[paper]](https://www.sciencedirect.com/science/article/pii/S0167865517302155)
2. **Cross-type biomedical named entity recognition with deep multi-task learning**. Wang X, Zhang Y, Ren X, et al. *Bioinformatics*, 2019, 35(10): 1745-1752. [[paper]](https://academic.oup.com/bioinformatics/article/35/10/1745/5126922) [[code]](https://github.com/yuzhimanhua/lm-lstm-crf) 
3. **Improving Chemical Named Entity Recognition in Patents with Contextualized Word Embeddings**.  Zhai Z, Nguyen D Q, Akhondi S, et al. *Proceedings of the 18th BioNLP Workshop and Shared Task*. 2019: 328-338. [[paper]](https://www.aclweb.org/anthology/W19-5035) [[code]](https://github.com/zenanz/ChemPatentEmbeddings)
4. **Chinese Clinical Named Entity Recognition Using Residual Dilated Convolutional Neural Network with Conditional Random Field**. Qiu J, Zhou Y, Wang Q, et al. *IEEE Transactions on NanoBioscience*, 2019, 18(3): 306-315. [[paper]](https://ieeexplore.ieee.org/abstract/document/8678833/)
5. **A Neural Multi-Task Learning Framework to Jointly Model Medical Named Entity Recognition and Normalization**. Zhao S, Liu T, Zhao S, et al. *Proceedings of the AAAI Conference on Artificial Intelligence*. 2019, 33: 817-824. [[paper]](https://wvvw.aaai.org/ojs/index.php/AAAI/article/download/3861/3739)
6. **CollaboNet: collaboration of deep neural networks for biomedical named entity recognition**. Yoon W, So C H, Lee J, et al. *BMC bioinformatics*, 2019, 20(10): 249. [[paper]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2813-6) [[code]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2813-6)
6. **BioBERT: a pre-trained biomedical language representation model for biomedical text mining**. Lee J, Yoon W, Kim S, et al. *Bioinformatics*, Advance article, 2019. [[paper]](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btz682/5566506) [[code]](https://github.com/dmis-lab/biobert)
7. **HUNER: Improving Biomedical NER with Pretraining**. Weber L, Münchmeyer J, Rocktäschel T, et al. *Bioinformatics*, Advance article, 2019. [[paper]](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btz528/5523847?redirectedFrom=fulltext) [[code]](https://hu-ner.github.io/)

- **Others**
1. **TaggerOne: joint named entity recognition and normalization with semi-Markov Models**. Leaman R, Lu Z. *Bioinformatics*, 2016, 32(18): 2839-2846. [[paper]](https://academic.oup.com/bioinformatics/article/32/18/2839/1744190) [[code]](https://www.ncbi.nlm.nih.gov/research/bionlp/tools/taggerone/)
2. **A transition-based joint model for disease named entity recognition and normalization**. Lou Y, Zhang Y, Qian T, et al. *Bioinformatics*, 2017, 33(15): 2363-2371. [[paper]](https://academic.oup.com/bioinformatics/article-abstract/33/15/2363/3089942) [[code]](https://github.com/louyinxia/jointRN)

