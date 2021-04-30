# -*- encoding: utf-8 -*-
'''
@File   :   model.py
@Time   :   2021/04/29 10:19:41
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   各种计算文本相似的方法
'''

import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessingPool as Pathos_Pool
from sentence_transformers import SentenceTransformer, util
import torch


#=========================================================
# EditDistance (编辑距离)
#=========================================================

class EditDistance:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        
    def _calc_distance(self, sent1, sent2):
        """
        如果sent1为sent2的子串，则相似度为f(x, y) = 1
        否则相似度f(x, y) = 1 - edit_distance(x,y) / max(len(x), len(y))
        """
        if sent1 in sent2:
            return 1.0

        len1 = len(sent1)
        len2 = len(sent2)
        dp = np.zeros((len1+1, len2+1))
        # dp = [[i + j for j in range(len(sent2)+1)] for i in range(len(sent1)+1)]
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                if sent1[i-1] == sent2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
        max_len = max(len1, len2)
        return 1 - dp[-1][-1] / max_len

    def get_top_n(self, query, corpus, n=5):
        # scores = np.zeros(self.corpus_size)
        temp_corpus = [" ".join(ele) for ele in corpus]
        pool = Pathos_Pool(cpu_count())
        scores = pool.map(self._calc_distance, [query]*self.corpus_size, temp_corpus)
        pool.close()
        scores = np.array(scores)
        # for i, sent in enumerate(tqdm(corpus)):
        #     scores[i] = self._calc_distance(query, sent)

        top_n = np.argsort(scores)[::-1][:n]
        return [{"idx": i, "document": corpus[i]} for i in top_n]


#=========================================================
# BM25 (Elastic Search默认方法)
#=========================================================

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, corpus, n=5):

        assert self.corpus_size == len(corpus), "The corpus given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [{"idx": i, "document": corpus[i]} for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


#=========================================================
# Sentence Transformers (基于预训练模型的语义相似度计算)
#=========================================================

class SentTranformers:
    def __init__(self, model_name):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        
    def _encode_sentences(self, corpus):
        """编码句子"""
        embeddings = self.encoder.encode(corpus, convert_to_tensor=True)
        return embeddings

    def _calc_similarity(self, embeddings1, embeddings2):
        """计算embedding之间的余弦相似度"""
        scores = util.pytorch_cos_sim(embeddings1, embeddings2)[0]
        return scores

    def get_top_n(self, query, corpus, n=5):
        temp_corpus = [" ".join(ele) for ele in corpus]
        pool = Pathos_Pool(cpu_count())
        corpus_embeddings = self._encode_sentences(temp_corpus)
        query_embeddings = self._encode_sentences(query)
        scores = self._calc_similarity(query_embeddings, corpus_embeddings)
        pool.close()
        top_results = torch.topk(scores, n)
        return [{"idx": i, "document": corpus[i]} for i in top_results[1].numpy().tolist()]