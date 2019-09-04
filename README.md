# NLP-zoo
Nlp related resources


## 中文语料（按用途进行分类）
对中文的NLP语料做个汇总，以备不时之需

### 闲聊常用语料

| 语料名称 | 语料Size | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [中文对白语料 chinese conversation corpus](https://github.com/fate233/dgk_lost_conv) | 单元格 | 单元格 | 可以用作聊天机器人的训练语料 |
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

### 领域特定语料

| 语料名称 | 语料大小 | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [保险行业QA语料库](https://github.com/Samurais/insuranceqa-corpus-zh)  | 未知 | 通过翻译 insuranceQA产生的数据集 | train_data含有问题12,889条，数据 141779条，正例：负例 = 1:10； test_data含有问题2,000条，数据 22000条，正例：负例 = 1:10；valid_data含有问题2,000条，数据 22000条，正例：负例 = 1:10 |
| [翻译语料(translation2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/translation2019zh.zip) | 520万个中英文平行语料( 原始数据1.1G，压缩文件596M) | 单元格 | 中英文平行语料520万对 |

### 序列标注语料

| 语料名称 | 语料大小 | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [微博实体识别](https://github.com/hltcoe/golden-horse) | 未知 | 未知 | 未知 |
| [boson数据](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/boson) | 未知 | 未知 | 包含6种实体类型 |
| [人民日报数据集](https://pan.baidu.com/s/1LDwQjoj7qc-HT9qwhJ3rcA) | 未知 | 未知 | password: 1fa3 |
| [MSRA微软亚洲研究院数据集](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA) | 未知 | 未知 | 5 万多条中文命名实体识别标注数据（包括地点、机构、人物） |

### 情感/观点/评论 倾向性分析

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

### 推荐系统

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| ez_douban | 5 万多部电影（3 万多有电影名称，2 万多没有电影名称），2.8 万 用户，280 万条评分数据 | [点击查看](./datasets/ez_douban/intro.ipynb) |
| dmsc_v2 | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 | [点击查看](./datasets/dmsc_v2/intro.ipynb) |
| yf_dianping | 24 万家餐馆，54 万用户，440 万条评论/评分数据 | [点击查看](./datasets/yf_dianping/intro.ipynb) |
| yf_amazon | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | [点击查看](./datasets/yf_amazon/intro.ipynb) |

### FAQ 问答系统

| 数据集 | 数据概览 | 下载地址 |
| ----- | -------- | ------- |
| 保险知道 | 8000 多条保险行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/baoxianzhidao/intro.ipynb) |
| 安徽电信知道 | 15.6 万条电信问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/anhuidianxinzhidao/intro.ipynb) |
| 金融知道 | 77 万条金融行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/financezhidao/intro.ipynb) |
| 法律知道 | 3.6 万条法律问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/lawzhidao/intro.ipynb) |
| 联通知道 | 20.3 万条联通问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/liantongzhidao/intro.ipynb) |
| 农行知道 | 4 万条农业银行问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/nonghangzhidao/intro.ipynb) |
| 保险知道 | 58.8 万条保险行业问答数据，包括用户提问、网友回答、最佳回答 | [点击查看](./datasets/baoxianzhidao/intro.ipynb) |

### 超大型通用语料

| 语料名称 | 语料大小 | 语料来源 | 语料描述 |
| :-----| ----: | :----: | :----: |
| [维基百科json版(wiki2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/wiki_zh_2019.zip) | 104万个词条, 1.6G | wiki | 做预训练的语料或构建词向量，也可以用于构建知识问答 |
| [新闻语料json版(news2016zh)](https://pan.baidu.com/s/1LJeq1dkA0wmYd9ZGZw72Xg) | 250万篇新闻,原始数据9G | 涵盖了6.3万个媒体，含标题、关键词、描述、正文 | 密码: film 包含了250万篇新闻。数据集划分：数据去重并分成三个部分。训练集：243万；验证集：7.7万；测试集，数万 |
| [百科类问答json版(baike2018qa)](https://pan.baidu.com/s/12TCEwC_Q3He65HtPKN17cA) | 150万个问答,原始数据1G多 | 密码:fu45 | 含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别 |
| [社区问答json版(webtext2019zh)](https://storage.googleapis.com/nlp_chinese_corpus/webtext2019zh.zip) | 410万个问答,过滤后数据3.7G | 1400万个原始问答 | 含有410万个预先过滤过的、高质量问题和回复。 |
| 单元格 | 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 | 单元格 |

### [中文同义词表，反义词表，否定词表](https://github.com/guotong1988/chinese_dictionary)

### 腾讯词向量
腾讯AI实验室公开的中文词向量数据集包含800多万中文词汇，其中每个词对应一个200维的向量。

下载地址：https://ai.tencent.com/ailab/nlp/embedding.html



# Paper4NLP

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

1. [LUO, Ling, et al. An attention-based BiLSTM-CRF approach to document-level chemical named entity recognition. Bioinformatics, 2017, 1: 8.](https://sci-hub.org.cn/hubdownload?s=https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btx761/23048295/btx761.pdf)

2. [TSAI, Richard Tzong-Han; HSIAO, Yu-Cheng; LAI, Po-Ting. NERChem: adapting NERBio to chemical patents via full-token features and named entity feature with chemical sub-class composition. Database, 2016, 2016.](https://academic.oup.com/database/article/doi/10.1093/database/baw135/2630527)

3. [DAI, Hong-Jie, et al. Enhancing of chemical compound and drug name recognition using representative tag scheme and fine-grained tokenization. Journal of cheminformatics, 2015, 7.S1: S14.
](https://link.springer.com/article/10.1186/1758-2946-7-S1-S14)

4. [HE, Hua, et al. An Insight Extraction System on BioMedical Literature with Deep Neural Networks. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. p. 2691-2701.](http://www.aclweb.org/anthology/D17-1285)

5. [CRICHTON, Gamal, et al. A neural network multi-task learning approach to biomedical named entity recognition. BMC bioinformatics, 2017, 18.1: 368.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8)

6. [李丽双; 郭元凯. 基于 CNN-BLSTM-CRF 模型的生物医学命名实体识别. 中文信息学报, 32.1: 116-122.](http://jcip.cipsc.org.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=2505)

7. [CHOI, Youngduck; CHIU, Chill Yi-I.; SONTAG, David. Learning low-dimensional representations of medical concepts. AMIA Summits on Translational Science Proceedings, 2016, 2016: 41.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5001761/)

8. [SMITH, Lawrence H., et al. MedTag: a collection of biomedical annotations. In: Proceedings of the ACL-ISMB workshop on linking biological literature, ontologies and databases: mining biological semantics. Association for Computational Linguistics, 2005. p. 32-37.](https://aclanthology.info/pdf/W/W05/W05-1305.pdf)

9. [ZHAO, Zhehuan, et al. ML-CNN: A novel deep learning based disease named entity recognition architecture. In: Bioinformatics and Biomedicine (BIBM), 2016 IEEE International Conference on. IEEE, 2016. p. 794-794.](https://sci-hub.org.cn/hubdownload?s=http://ieeexplore.ieee.org/abstract/document/7822625/)

10. [KANIMOZHI, U.; MANJULA, D. A CRF Based Machine Learning Approach for Biomedical Named Entity Recognition. In: Recent Trends and Challenges in Computational Models (ICRTCCM), 2017 Second International Conference on. IEEE, 2017. p. 335-342.](https://sci-hub.org.cn/hubdownload?s=http://ieeexplore.ieee.org/abstract/document/8057560/)

11. [REI, Marek; CRICHTON, Gamal KO; PYYSALO, Sampo. Attending to characters in neural sequence labeling models. arXiv preprint arXiv:1611.04361, 2016.
](https://arxiv.org/pdf/1611.04361)

12. [MURUGESAN, Gurusamy, et al. BCC-NER: bidirectional, contextual clues named entity tagger for gene/protein mention recognition. EURASIP Journal on Bioinformatics and Systems Biology, 2017, 2017.1: 7.](https://bsb-eurasipjournals.springeropen.com/articles/10.1186/s13637-017-0060-6)

13. [AL-HEGAMI, Ahmed Sultan; OTHMAN, Ameen Mohammed Farea; BAGASH, Fuad Tarbosh. A biomedical named entity recognition using machine learning classifiers and rich feature set. International Journal of Computer Science and Network Security (IJCSNS), 2017, 17.1: 170.](http://paper.ijcsns.org/07_book/201701/20170126.pdf)

14. [CHO, Hyejin; CHOI, Wonjun; LEE, Hyunju. A method for named entity normalization in biomedical articles: application to diseases and plants. BMC bioinformatics, 2017, 18.1: 451.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1857-8)

15. [LI, Haodi, et al. CNN-based ranking for biomedical entity normalization. BMC bioinformatics, 2017, 18.11: 385.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1805-7)

16. [Disease named entity recognition by combining conditional random fields and bidirectional recurrent neural networks](https://www.researchgate.net/profile/Ruifeng_Xu2/publication/309719928_Disease_named_entity_recognition_by_combining_conditional_random_fields_and_bidirectional_recurrent_neural_networks/links/5825e7c608aeb45b5892c953/Disease-named-entity-recognition-by-combining-conditional-random-fields-and-bidirectional-recurrent-neural-networks.pdf)

17. [PENG, Yifan; LU, Zhiyong. Deep learning for extracting protein-protein interactions from biomedical literature. arXiv preprint arXiv:1706.01556, 2017.](https://arxiv.org/pdf/1706.01556)

18. [LOU, Yinxia, et al. A transition-based joint model for disease named entity recognition and normalization. Bioinformatics, 2017, 33.15: 2363-2371.](https://sci-hub.org.cn/hubdownload?s=https://academic.oup.com/bioinformatics/article/3089942)

19. [PENG, Yifan; LU, Zhiyong. Deep learning for extracting protein-protein interactions from biomedical literature. arXiv preprint arXiv:1706.01556, 2017.](https://arxiv.org/pdf/1706.01556)

20. [LUO, Ling, et al. DUTIR at the BioCreative V. 5. BeCalm Tasks: A BLSTM-CRF Approach for Biomedical Entity Recognition in Patents.](https://www.researchgate.net/profile/Ling_Luo11/publication/317060280_DUTIR_at_the_BioCreative_V5BeCalm_Tasks_A_BLSTM-CRF_Approach_for_Biomedical_Entity_Recognition_in_Patents/links/59258b15458515e3d44581c8/DUTIR-at-the-BioCreative-V5BeCalm-Tasks-A-BLSTM-CRF-Approach-for-Biomedical-Entity-Recognition-in-Patents.pdf)
