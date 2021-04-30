# -*- encoding: utf-8 -*-
'''
@File   :   similarity.py
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   文本相似度计算 (主函数)
'''

from utils import load_data
from model import BM25Okapi
from model import EditDistance
from model import SentTranformers


def get_data(in_path):
    data = load_data(in_path)
    
    idx2number = {}
    number2data = {}
    corpus = []  # 用来计算文本相似度的文档集
    for idx, sample in enumerate(data):
        number = sample["number"]
        product = sample["product"]
        component = sample["component"]
        abstract = sample["abstract"]
        description = sample["description"]
        recreation_procedure = sample["recreation_procedure"]
        problem_isolation = sample["problem_isolation"]

        full_data = {"product": product,
                     "component": component,
                     "abstract": abstract,
                     "description": description}
        
        idx2number[idx] = number
        number2data[number] = full_data
        
        # 根据需求获取不同字段的数据
        sentence = sample["abstract"] if sample["abstract"] else sample["description"]
        tokenized_sent = sentence.split(" ")  # 这里可以写一些自己需求的清洗函数
        corpus.append(tokenized_sent)

    return idx2number, number2data, corpus

        
if __name__ == "__main__":

    print("正在加载数据......")
    idx2number, number2data, corpus = get_data("data/processed_data.json")

    # corpus = corpus[:100]
 
    #=======================
    # EditDistance(不适合英文句子，适合英文单词；中文句子适合)/BM25
    #=======================
    print("模型初始化中......")
    ed = EditDistance(corpus)
    # bm25 = BM25Okapi(corpus)
    embedder = SentTranformers("paraphrase-distilroberta-base-v1")
    
    for _ in range(1):
        print("---------------------------------")
        # EditDistance
        # ed_query = input("请输入查询：")
        # return_data = ed.get_top_n(ed_query, corpus, n=5)

        # BM25
        # bm25_tokenized_query = input("请输入查询：").split(" ")
        # return_data = bm25.get_top_n(bm25_tokenized_query, corpus, n=5)

        # Sentence Transformer
        st_query = input("请输入查询：")
        return_data = embedder.get_top_n([st_query], corpus)

        for ele in return_data:
            number = idx2number[ele["idx"]]
            similarity_data = " ".join(ele["document"])
            # similarity_full_data = number2data[number]
            
            print("number: ", number)
            print("similarity_data: ", similarity_data)
            print("\n")
            


    

