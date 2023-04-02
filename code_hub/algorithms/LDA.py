# _*_ coding:utf-8 _*_
# @Time: 2023-04-02 18:45
# @Author: peters
# @Email: xinbao.sun@hotmail.com
# @File: LDA.py
# @Project: CommonCodes

def lda_analysis(txts, topics, words, parallel=True, use_tfidf=True):
    """
    txts: 文本列表
    topics: 聚类主题数量
    parallel: 并行化
    use_tfidf: 是否使用tfidf执行分析
    """
    from gensim import models
    from gensim.models.ldamodel import LdaModel
    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary
    dictionary = Dictionary(txts)
    texts = [dictionary.doc2bow(text) for text in txts]
    texts_tf_idf = TfidfModel(texts)[texts]
    if not parallel:
        if not use_tfidf:
            lda = LdaModel(corpus=texts, id2word=dictionary, num_topics=topics)
        else:
            lda = LdaModel(corpus=texts_tf_idf, id2word=dictionary, num_topics=topics)
    else:
        lda = models.ldamulticore.LdaMulticore(corpus=texts_tf_idf, id2word=dictionary, num_topics=topics)
    return lda.show_topics(topics,num_words=words)