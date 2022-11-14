# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 8:42 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""
import os
import sys
import numpy as np
from itertools import combinations

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)


def concept_mining(name_kg):
    from hub.concept_id import create_id
    # 同义词子图挖掘，同一概念create一个唯一ID
    # name_kg {word: [synonyms]}
    import networkx as nx
    g = nx.Graph()
    for name in name_kg:
        if not g.has_node(name):
            g.add_node(name)
        for name2 in name_kg[name]:
            if not g.has_node(name2):
                g.add_node(name2)
            g.add_edge(name, name2)
    concept_id_dict = {}
    for c in nx.connected_components(g):
        sub_g = g.subgraph(c)
        node_set = sub_g.nodes()
        nodes = [str(node) for node in node_set]
        concept_id = create_id()
        concept_id_dict[concept_id] = nodes

    return concept_id_dict

def _single_pass(cls, iterator, threshold=0.6):
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

        class Accumulator:  # @save
            """在`n`个变量上累加。"""

            def __init__(self, n):
                self.data = [0.0] * n

            def add(self, *args):
                self.data = [a + float(b) for a, b in zip(self.data, args)]

            def reset(self):
                self.data = [0.0] * len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        class LshTraditional(object):
            @classmethod
            def build_shingles(cls, sentence: str, k: int):
                shingles = []
                for i in range(len(sentence) - k):
                    shingles.append(sentence[i:i + k])
                return set(shingles)

            @classmethod
            def build_vocab(cls, shingle_sets: list):
                # convert list of shingle sets into single set
                full_set = {item for set_ in shingle_sets for item in set_}
                vocab = {}
                for i, shingle in enumerate(list(full_set)):
                    vocab[shingle] = i
                return vocab

            @classmethod
            def one_hot(cls, shingles: set, vocab: dict):
                vec = np.zeros(len(vocab))
                for shingle in shingles:
                    idx = vocab[shingle]
                    vec[idx] = 1
                return vec

            @classmethod
            def make_k_shingles(cls, sentences):
                k = 8  # shingle size

                # build shingles
                shingles = []
                for sentence in sentences:
                    shingles.append(cls.build_shingles(sentence, k))

                # build vocab
                vocab = cls.build_vocab(shingles)

                # one-hot encode our shingles
                shingles_1hot = []
                for shingle_set in shingles:
                    shingles_1hot.append(cls.one_hot(shingle_set, vocab))
                # stack into single numpy array
                shingles_1hot = np.stack(shingles_1hot)
                return shingles_1hot, vocab

            @classmethod
            def minhash_funcs_arr(cls, vocab: dict, resolution: int):
                length = len(vocab.keys())
                arr = np.zeros((resolution, length))
                for i in range(resolution):
                    permutation = np.random.permutation(len(vocab)) + 1
                    arr[i, :] = permutation.copy()
                return arr.astype(int)

    @classmethod
    def get_signature(cls, minhash_funcs, vector):
        # get index locations of every 1 value in vector
        idx = np.nonzero(vector)[0].tolist()
        # use index locations to pull only +ve positions in minhash
        shingles = minhash_funcs[:, idx]
        # find minimum value in each hash vector
        signature = np.min(shingles, axis=1)
        return signature

    @classmethod
    def make_subvecs(cls, signature, bands):
        l = len(signature)
        assert l % bands == 0
        r = int(l / bands)
        # break signature into subvectors
        subvecs = []
        for i in range(0, l, r):
            subvecs.append(signature[i:i + r])
        return np.stack(subvecs)

    def add_hash(self, signature, bands):
        subvecs = self.make_subvecs(signature, bands).astype(str)
        for i, subvec in enumerate(subvecs):
            subvec = ','.join(subvec)
            if subvec not in self.buckets[i].keys():
                self.buckets[i][subvec] = []
            self.buckets[i][subvec].append(self.counter)
        self.counter += 1

    def check_candidates(self):
        candidates = []
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hits = bucket_band[bucket]
                if len(hits) > 1:
                    candidates.extend(combinations(hits, 2))
        return set(candidates)

    def process(self, sentences, resolution, bands):
        """

        :param sentences:
        :param resolution: 降维维度
        :return:
        """
        self.buckets = []
        self.counter = 0
        shingles_1hot, vocab = self.make_k_shingles(sentences)
        minhash_funcs = self.minhash_funcs_arr(vocab, resolution)
        for vector in shingles_1hot:
            signature = self.get_signature(minhash_funcs, vector)
            self.add_hash(signature, bands)


class Lsh:
    pass


if __name__ == '__main__':
    pass