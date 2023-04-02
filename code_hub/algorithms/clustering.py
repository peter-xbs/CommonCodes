# _*_ coding:utf-8 _*_
# @Time: 2023-04-02 18:42
# @Author: peters
# @Email: xinbao.sun@hotmail.com
# @File: clustering.py
# @Project: CommonCodes

def _single_pass(cls, iterator, threshold=0.6):
    """
    自定义实现，效率较低
    """
    import numpy as np
    def getMaxSimilarity(dictTopic, template):
        maxValue = 0
        maxIndex = -1
        for k, cluster in dictTopic.items():
            oneSimilarity = np.mean([cls._jaccard_sim(template, vector) for vector in cluster])
            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    dictTopic = {}
    topic2idx = {}  # 存储template2content 的索引
    numTopic = 0
    cnt = 0
    for idx, template in enumerate(iterator):
        template_str = str(template)
        if numTopic == 0:
            dictTopic[numTopic] = []
            dictTopic[numTopic].append(template)

            topic2idx[numTopic] = {}
            topic2idx[numTopic][template_str] = []
            topic2idx[numTopic][template_str].append(idx)
            numTopic += 1
        else:
            maxIndex, maxValue = getMaxSimilarity(dictTopic, template)
            # join the most similar topic
            if maxValue > threshold:
                dictTopic[maxIndex].append(template)
                if template_str not in topic2idx[maxIndex]:
                    topic2idx[maxIndex][template_str] = []
                topic2idx[maxIndex][template_str].append(idx)
            # else create the new topic
            else:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(template)

                topic2idx[numTopic] = {}
                topic2idx[numTopic][template_str] = []
                topic2idx[numTopic][template_str].append(idx)
                numTopic += 1
        cnt += 1
        if cnt % 1000 == 0:
            print("processing {}".format(cnt))
    return dictTopic, topic2idx


def _single_pass2(iterator, threshold):
    """
    借助gensim实现，主要基于tf或者tf_idf来实现
    """
    from gensim.similarities import MatrixSimilarity
    from gensim.corpora import Dictionary
    # from gensim.models import TfidfModel
    texts = iterator
    dictionary = Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    # tfidf = TfidfModel(corpus)

    # corpus_tfidf = tfidf[corpus] #该行注释掉后，表明仅用词频

    # index = MatrixSimilarity(corpus_tfidf)
    # cos_similarity = [list(index[vector]) for vector in corpus_tfidf]
    index = MatrixSimilarity(corpus)
    cos_similarity = [list(index[vector]) for vector in corpus]
    processed = [(0, 0)]
    cluster_map = {0: [0]}  # 存储文本簇
    cluster_id = 1
    for i in range(1, len(cos_similarity)):
        cos_list = cos_similarity[i][0:i]
        max_similarity = max(cos_list)
        if max_similarity > threshold:
            max_similarity_index = cos_list.index(max_similarity)
            related_cluter_id = processed[max_similarity_index][1]
            processed.append((i, related_cluter_id))
            cluster_map[related_cluter_id].append(i)
        else:
            processed.append((i, cluster_id))
            cluster_map[cluster_id] = [i]
            cluster_id += 1

        return cluster_map

def single_pass3(vectors, threshold):
    """
    借助sklearn来实现，可直接输入一组向量来进行聚类
    """
    from sklearn.metrics.pairwise import cosine_similarity
    cos_similarity = cosine_similarity(vectors).tolist()
    processed = [(0, 0)]
    cluster_map = {0: [0]}  # 存储文本簇
    cluster_id = 1
    for i in range(1, len(cos_similarity)):
        cos_list = cos_similarity[i][0:i]
        max_similarity = max(cos_list)
        if max_similarity > threshold:
            max_similarity_index = cos_list.index(max_similarity)
            related_cluter_id = processed[max_similarity_index][1]
            processed.append((i, related_cluter_id))
            cluster_map[related_cluter_id].append(i)
        else:
            processed.append((i, cluster_id))
            cluster_map[cluster_id] = [i]
            cluster_id += 1
    return cluster_map


def bagging_cluster(txts, embs, k, methods=['km', 'af', 'db', 'ag']):
    """
    集合4种常见聚类方式，txt为文本列表，emb为每个文本对应的向量
    """
    def kmeans_cluster(X, k):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        return kmeans.labels_

    def affinity_cluster(X, k):
        from sklearn.cluster import AffinityPropagation
        clustering = AffinityPropagation(random_state=5).fit(X)
        return clustering.labels_

    def dbscan_cluster(X, k):
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.8, metric='cosine').fit(X)
        return clustering.labels_

    def agglomerative_cluster(X, k):
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=k).fit(X)
        return clustering.labels_

    cols = ['prompts']+ methods
    data = [txts]
    if 'km' in methods:
        km =kmeans_cluster(embs, k)
        data.append(km)
    if 'af' in methods:
        af = affinity_cluster(embs, k)
        data.append(af)
    if 'db' in methods:
        db = dbscan_cluster(embs, k)
        data.append(db)
    if 'ag' in methods:
        ag = agglomerative_cluster(embs, k)
        data.append(ag)
    lines = list(zip(*data))
    df = pd.DataFrame(lines, columns=cols)
    return df