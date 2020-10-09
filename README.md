# NLP-zoo

> 对中文的NLP资源做个汇总，以备不时之需


# 目录

* [中文NLP语料资源](#中文NLP语料资源)
* [NLP工具包](#NLP工具包)
* [中文分词](#中文分词)
* [QA & Chatbot 问答和聊天机器人](#Chatbot问答和聊天机器人)

* [NLP相关论文](#NLP相关论文)
* [Python-Tutorial](#Python-Tutorial)
* [matplotlib-Ipython](#matplotlib-Ipython)
* [pandas](#pandas)
* [scikit-learn](#scikit-learn)



# 中文NLP语料资源

> https://mp.weixin.qq.com/s/CODsntmNYwHga9jHxDdbgw

## 闲聊

| 语料名称                                                     | 语料Size |            语料来源             |               语料描述                |
| :----------------------------------------------------------- | -------: | :-----------------------------: | :-----------------------------------: |
| [中文对白语料 chinese conversation corpus](https://github.com/fate233/dgk_lost_conv) |          |                                 |     可以用作聊天机器人的训练语料      |
| [chatterbot](https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese) |      560 |            开源项目             |         按类型分类，质量较高          |
| qingyun（青云语料）                                          |      10W |       某聊天机器人交流群        |           相对不错，生活化            |
| [xiaohuangji（小黄鸡语料）](https://github.com/candlewill/Dialog_Corpus) |      45W |        原人人网项目语料         |       有一些不雅对话，少量噪音        |
| [douban（豆瓣多轮）](https://github.com/MarkWuNLP/MultiTurnResponseSelection) |     352W | 来自北航和微软的paper, 开源项目 | 噪音相对较少，原本是多轮（平均7.6轮） |
| [weibo（微博语料）]()                                        |     443W |         来自华为的paper         |              有一些噪音               |

```
使用方法
下载语料 https://pan.baidu.com/s/1szmNZQrwh9y994uO8DFL_A 提取码：f2ex

执行命令即可
python main.py

生成结果
每个来源的语料分别生成一个独立的*.tsv文件，都放在新生成的clean_chat_corpus文件夹下。
生成结果格式为 tsv格式，每行是一个样本，先是query，再是answer
```

## 领域特定语料

| 语料名称                                                     |                                           语料大小 |             语料来源             |                           语料描述                           |
| :----------------------------------------------------------- | -------------------------------------------------: | :------------------------------: | :----------------------------------------------------------: |
| [保险行业QA语料库](https://github.com/Samurais/insuranceqa-corpus-zh) |                                               未知 | 通过翻译 insuranceQA产生的数据集 | train_data含有问题12,889条，数据 141779条，正例：负例 = 1:10； test_data含有问题2,000条，数据 22000条，正例：负例 = 1:10；valid_data含有问题2,000条，数据 22000条，正例：负例 = 1:10 |
| [翻译语料(translation2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/translation2019zh.zip) | 520万个中英文平行语料( 原始数据1.1G，压缩文件596M) |              单元格              |                    中英文平行语料520万对                     |

## NER

| 语料名称                                                     |            语料大小 |                     语料来源                      |                           语料描述                           |
| :----------------------------------------------------------- | ------------------: | :-----------------------------------------------: | :----------------------------------------------------------: |
| [weibo NER corpus](https://github.com/hltcoe/golden-horse)   |                未知 |                       未知                        | 包含了1,890条，设计的实体有：人名、地点、组织、地理政治相关实体 |
| [boson数据(不维护了)](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/boson) |              2000条 |                       未知                        |  包含人名、地名、时间、组织名、公司名、产品名这6种实体类型   |
| [1998人民日报](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily) | 新闻一共有137万多条 |                       未知                        |                  包含地名、人名和机构名三类                  |
| [MSRA](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA) |                未知 |                       未知                        | 5 万多条中文命名实体识别标注数据（IOB2 格式，符合 CoNLL 2002 和 CRF++ 标准）包含地名、人名和机构名三类 |
| [Resume NER data](https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER) |                     | ACL 2018 paper 《Chinese NER Using Lattice LSTM》 | 爬虫新浪财经的的简历数据, CoNLL format (BIOES tag scheme)，包括城市、学校、地点、人名、组织等 |
| [影视、音乐、书籍](https://github.com/LG-1/video_music_book_datasets) |                未知 |                       未知                        | 类似于人名/地名/组织机构名的命名体识别数据集，大约10000条影视/音乐/书籍数据 |
| [1300W字的新闻](https://pan.baidu.com/s/17djsvYfpYUXrazL0H_mtoA) |                未知 |                       未知                        | 该语料可用于分词、NER、POS等任务。标记和格式请参考此文章(https://cloud.tencent.com/developer/article/1091906) |
| [CCKS2017中文电子病例命名实体识别](https://biendata.com/competition/CCKS2017_2/data/) |                     |            北京极目云健康科技有限公司             | 数据来源于其云医院平台的真实电子病历数据，共计800条（单个病人单次就诊记录），经脱敏处理 |
| [CCKS2018中文电子病例命名实体识别](https://biendata.com/competition/CCKS2018_1/data/) |                     |            医渡云（北京）技术有限公司             | CCKS2018的电子病历命名实体识别的评测任务提供了600份标注好的电子病历文本，共需识别含解剖部位、独立症状、症状描述、手术和药物五类实体 |
| [CLUE Fine-Grain NER](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip) |                     |                       CLUE                        | CLUENER2020数据集，是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS。数据包含10个标签类别，训练集共有10748条语料，验证集共有1343条语料 |
|                                                              |                     |                                                   |                                                              |



## 文本分类

| 语料名称                                                     |   语料大小 | 语料来源 |                           语料描述                           |
| :----------------------------------------------------------- | ---------: | :------: | :----------------------------------------------------------: |
| [2018中国‘法研杯’法律智能挑战赛数据](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) |       未知 |   未知   | 268万刑法法律文书，共涉及183条罪名，202条法条，刑期长短包括0-25年、无期、死刑 |
| [今日头条中文新闻（短文本）](https://github.com/fateleak/toutiao-text-classfication-dataset) | 共382688条 |   未知   | 15个分类中，包含民生、文化、娱乐、体育、财经、房产、骑车、教育、科技、军事、旅游、国际、证券、农业、电竞 |
| [SMP2017中文人机对话评测数据](https://github.com/HITlilingzhi/SMP2017ECDT-DATA) |          - |   未知   | 包含了两个任务的数据集，用户意图领域分类和特定域任务型人机对话在线评测。第一个数据集用得比较多。用户意图领域分类包含闲聊类、任务垂直类共三十一个类别，属于短文本分类的一个范畴 |

## 情感/观点/评论 倾向性分析

| 数据集                  | 数据概览                                                     | 下载地址                                                   |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| ChnSentiCorp_htl_all    | 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论  | [点击查看](./datasets/ChnSentiCorp_htl_all/intro.ipynb)    |
| waimai_10k              | 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条      | [点击查看](./datasets/waimai_10k/intro.ipynb)              |
| online_shopping_10_cats | 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，<br /> 包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店 | [点击查看](./datasets/online_shopping_10_cats/intro.ipynb) |
| weibo_senti_100k        | 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条        | [点击查看](./datasets/weibo_senti_100k/intro.ipynb)        |
| simplifyweibo_4_moods   | 36 万多条，带情感标注 新浪微博，包含 4 种情感，<br /> 其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条 | [点击查看](./datasets/simplifyweibo_4_moods/intro.ipynb)   |
| dmsc_v2                 | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据         | [点击查看](./datasets/dmsc_v2/intro.ipynb)                 |
| yf_dianping             | 24 万家餐馆，54 万用户，440 万条评论/评分数据                | [点击查看](./datasets/yf_dianping/intro.ipynb)             |
| yf_amazon               | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb)               |

## 推荐系统

| 数据集      | 数据概览                                                     | 下载地址                                       |
| ----------- | ------------------------------------------------------------ | ---------------------------------------------- |
| ez_douban   | 5 万多部电影（3 万多有电影名称，2 万多没有电影名称），2.8 万 用户，280 万条评分数据 | [点击查看](./datasets/ez_douban/intro.ipynb)   |
| dmsc_v2     | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据         | [点击查看](./datasets/dmsc_v2/intro.ipynb)     |
| yf_dianping | 24 万家餐馆，54 万用户，440 万条评论/评分数据                | [点击查看](./datasets/yf_dianping/intro.ipynb) |
| yf_amazon   | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb)   |

## FAQ 问答

| 数据集                | 数据概览                                                     | 下载地址                                                     |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 保险知道              | 8000 多条保险行业问答数据，包括用户提问、网友回答、最佳回答  | [点击查看](./datasets/baoxianzhidao/intro.ipynb)             |
| 安徽电信知道          | 15.6 万条电信问答数据，包括用户提问、网友回答、最佳回答      | [点击查看](./datasets/anhuidianxinzhidao/intro.ipynb)        |
| 金融知道              | 77 万条金融行业问答数据，包括用户提问、网友回答、最佳回答    | [点击查看](./datasets/financezhidao/intro.ipynb)             |
| 法律知道              | 3.6 万条法律问答数据，包括用户提问、网友回答、最佳回答       | [点击查看](./datasets/lawzhidao/intro.ipynb)                 |
| 联通知道              | 20.3 万条联通问答数据，包括用户提问、网友回答、最佳回答      | [点击查看](./datasets/liantongzhidao/intro.ipynb)            |
| 农行知道              | 4 万条农业银行问答数据，包括用户提问、网友回答、最佳回答     | [点击查看](./datasets/nonghangzhidao/intro.ipynb)            |
| 保险知道              | 58.8 万条保险行业问答数据，包括用户提问、网友回答、最佳回答  | [点击查看](./datasets/baoxianzhidao/intro.ipynb)             |
| 580万百度知道社群问答 | 包括超过580万的问题，每个问题带有问题标签。问答对983万个，每个问题的答案个数1.7个，问题标签个数5824个。 | [点击查看](https://github.com/liuhuanyong/MiningZhiDaoQACorpus) |
| DuReader              | 百度开源的一个QA和MRC数据集，共140万篇文档，30万个问题，及66万个答案。 | [点击查看](http://ai.baidu.com/broad/introduction?dataset=dureader) |
| 社区问答数据          | 含有410万个预先过滤过的、高质量问题和回复。每个问题属于一个话题，总共有2.8万个各式话题，话题包罗万象。从1400万个原始问答中，筛选出至少获得3个点赞以上的的答案，代表了回复的内容比较不错或有趣，从而获得高质量的数据集。除了对每个问题对应一个话题、问题的描述、一个或多个回复外，每个回复还带有点赞数、回复ID、回复者的标签 | [点击查看](https://github.com/brightmart/nlp_chinese_corpus) |

## 最新任务型对话数据集大全

> 由哈工大SCIR博士生侯宇泰收集整理的一个任务型对话数据集大全

这份数据集大全涵盖了到目前在任务型对话领域的所有常用数据集的主要信息。
此外，为了帮助研究者更好的把握领域进展的脉络，以Leaderboard的形式给出了几个数据集上的State-of-the-art实验结果。

数据集的地址如下：

https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey

## 超大型通用语料

| 语料名称                                                     |                   语料大小 |                   语料来源                    |                           语料描述                           |
| :----------------------------------------------------------- | -------------------------: | :-------------------------------------------: | :----------------------------------------------------------: |
| [维基百科json版(wiki2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/wiki_zh_2019.zip) |          104万个词条, 1.6G |                     wiki                      |      做预训练的语料或构建词向量，也可以用于构建知识问答      |
| [新闻语料json版(news2016zh)](https://pan.baidu.com/s/1LJeq1dkA0wmYd9ZGZw72Xg) |     250万篇新闻,原始数据9G | 涵盖了6.3万个媒体，含标题、关键词、描述、正文 | 密码: film 包含了250万篇新闻。数据集划分：数据去重并分成三个部分。训练集：243万；验证集：7.7万；测试集，数万 |
| [百科类问答json版(baike2018qa)](https://pan.baidu.com/s/12TCEwC_Q3He65HtPKN17cA) |   150万个问答,原始数据1G多 |                   密码:fu45                   | 含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别 |
| [社区问答json版(webtext2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/webtext2019zh.zip) | 410万个问答,过滤后数据3.7G |               1400万个原始问答                |         含有410万个预先过滤过的、高质量问题和回复。          |

## 其他资源


- **[中文同义词表，反义词表，否定词表](https://github.com/guotong1988/chinese_dictionary)**
- 1.4亿三元组中文知识图谱（https://github.com/ownthink/KnowledgeGraphData）
- Dbpedia（https://wiki.dbpedia.org/develop/datasets/dbpedia-version-2016-10）：多语知识图谱数据，共有130亿个三元组，但大部分都是英语。有760个类，1105个关系，1622个属性。
- 开放的中文知识图谱社区（http://www.openkg.cn/）：这里有很多垂直领域图谱数据，我就不一一放上来了。
- **腾讯词向量**

  - 腾讯AI实验室公开的中文词向量数据集包含800多万中文词汇，其中每个词对应一个200维的向量。

  - 下载地址：https://ai.tencent.com/ailab/nlp/embedding.html



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

  

# Chatbot问答和聊天机器人

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

## Attention

1. [SEO, Minjoon, et al. Bidirectional attention flow for machine comprehension. arXiv preprint arXiv:1611.01603, 2016.](https://arxiv.org/pdf/1611.01603)

2. [ZADEH, Amir, et al. Multi-attention recurrent network for human communication comprehension. arXiv preprint arXiv:1802.00923, 2018.](https://arxiv.org/pdf/1802.00923)

3. [CHEN, Kehai, et al. Syntax-Directed Attention for Neural Machine Translation. arXiv preprint arXiv:1711.04231, 2017.](https://arxiv.org/pdf/1711.04231)

## 词向量表示学习

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


## 对抗GAN

1. [王坤峰, et al. 生成式对抗网络 GAN 的研究进展与展望. 自动化学报, 2017, 43.3: 321-332.](http://html.rhhz.net/ZDHXBZWB/html/20170301.htm)

2. [CHEN, Xinchi, et al. Adversarial multi-criteria learning for chinese word segmentation. arXiv preprint arXiv:1704.07556, 2017.](https://arxiv.org/pdf/1704.07556)

3. [LIU, Pengfei; QIU, Xipeng; HUANG, Xuanjing. Adversarial multi-task learning for text classification. arXiv preprint arXiv:1704.05742, 2017.](https://arxiv.org/pdf/1704.05742)

4. [LI, Zheng, et al. End-to-end adversarial memory network for cross-domain sentiment classification. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI). 2017. p. 2237.](https://www.ijcai.org/proceedings/2017/0311.https://arxiv.org/pdf/1710.07035pdf)

5. [GUI, Tao, et al. Part-of-speech tagging for twitter with adversarial neural networks. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2411-2420.](http://www.aclweb.org/anthology/D17-1256)

6. [KIM, Joo-Kyung, et al. Cross-Lingual Transfer Learning for POS Tagging without Cross-Lingual Resources. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2832-2838.](http://www.aclweb.org/anthology/D17-1302)

7. [CRESWELL, Antonia, et al. Generative Adversarial Networks: An Overview. IEEE Signal Processing Magazine, 2018, 35.1: 53-65.](https://arxiv.org/pdf/1710.07035)

## 多任务学习

1. [CRICHTON, Gamal, et al. A neural network multi-task learning approach to biomedical named entity recognition. BMC bioinformatics, 2017, 18.1: 368.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8)

2. [Chen, X., Qiu, X., & Huang, X. (2016). A feature-enriched neural model for joint Chinese word segmentation and part-of-speech tagging. arXiv preprint arXiv:1611.05384.](https://arxiv.org/pdf/1611.05384)

3. [RUDER, Sebastian. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.](https://arxiv.org/pdf/1706.05098)

4. [LONG, Mingsheng, et al. Learning Multiple Tasks with Multilinear Relationship Networks. In: Advances in Neural Information Processing Systems. 2017. p. 1593-1602.](http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-multilinear-relationship-networks.pdf)

5. [AGUILAR, Gustavo, et al. A Multi-task Approach for Named Entity Recognition in Social Media Data. In: Proceedings of the 3rd Workshop on Noisy User-generated Text. 2017. p. 148-153.](http://www.aclweb.org/anthology/W17-4419)


## 关系抽取任务

1. [WU, Yi; BAMMAN, David; RUSSELL, Stuart. Adversarial training for relation extraction. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 1778-1783.](http://www.aclweb.org/anthology/D17-1187)

2. [HUANG, Yi Yao; WANG, William Yang. Deep Residual Learning for Weakly-Supervised Relation Extraction. arXiv preprint arXiv:1707.08866, 2017.](https://arxiv.org/pdf/1707.08866)

3. [HUANG, Yi Yao; WANG, William Yang. Deep Residual Learning for Weakly-Supervised Relation Extraction. arXiv preprint arXiv:1707.08866, 2017.](http://www.aclweb.org/anthology/D17-1182)

4. [HE, Zhengqiu, et al. SEE: Syntax-aware Entity Embedding for Neural Relation Extraction. arXiv preprint arXiv:1801.03603, 2018.](https://arxiv.org/pdf/1801.03603)

5. [GANEA, Octavian-Eugen; HOFMANN, Thomas. Deep Joint Entity Disambiguation with Local Neural Attention. arXiv preprint arXiv:1704.04920, 2017.](https://arxiv.org/pdf/1704.04920)

6. [ADEL, Heike; SCHÜTZE, Hinrich. Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification. arXiv preprint arXiv:1707.07719, 2017.](https://arxiv.org/pdf/1707.07719)

7. [Zeng, W., Lin, Y., Liu, Z., & Sun, M. (2016). Incorporating relation paths in neural relation extraction. arXiv preprint arXiv:1609.07479.](https://arxiv.org/pdf/1609.07479)

8. [TAY, Yi; LUU, Anh Tuan; HUI, Siu Cheung. Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis. arXiv preprint arXiv:1712.05403, 2017.](https://arxiv.org/pdf/1712.05403)

9. [Zeng, X., He, S., Liu, K., & Zhao, J. (2018). Large Scaled Relation Extraction with Reinforcement Learning. Relation, 2, 3.](http://159.226.21.68/bitstream/173211/20626/1/Large%20Scaled%20Relation%20Extraction%20with%20Reinforcement%20Learning.pdf)


## 迁移学习

1. [KIM, Joo-Kyung, et al. Cross-Lingual Transfer Learning for POS Tagging without Cross-Lingual Resources. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2832-2838.](http://www.aclweb.org/anthology/D17-1302)

2. [YANG, Zhilin; SALAKHUTDINOV, Ruslan; COHEN, William W. Transfer learning for sequence tagging with hierarchical recurrent networks. arXiv preprint arXiv:1703.06345, 2017.](https://arxiv.org/pdf/1703.06345)

3. [PAN, Sinno Jialin; YANG, Qiang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 2010, 22.10: 1345-1359.](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)

4. [PAN, Sinno Jialin, et al. Domain adaptation via transfer component analysis. IEEE Transactions on Neural Networks, 2011, 22.2: 199-210.](http://www.aaai.org/ocs/index.php/IJCAI/IJCAI-09/paper/download/294/962)


## 情感分类 

1. [WANG, Bailin; LU, Wei. Learning Latent Opinions for Aspect-level Sentiment Classification. 2018.](http://www.statnlp.org/wp-content/uploads/papers/2018/Learning-Latent/absa.pdf)

