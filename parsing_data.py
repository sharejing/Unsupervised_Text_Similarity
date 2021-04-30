# -*- encoding: utf-8 -*-
'''
@File   :   parsing_data.py
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   从excel文件中抽取关键字段信息存入json文件
'''

import glob
import pandas as pd
import json


def read_file(in_path):
    return glob.glob(in_path)


def read_excel(filename, sheet_name="Sheet1"):
    # data = pd.read_excel(filename, sheet_name, index_col=["Name", "Component", "Description"])
    data = pd.read_excel(filename, sheet_name)
    return data



if __name__ == "__main__":

    corpus = []

    for filename in read_file("./data/*/*Information*.xlsx"):
        print(filename)
        print("\n")
        data = read_excel(filename)

        Description = data.get("Description", "")
        for number, name, component, description in zip(data["Number"], data["Name"], data["Component"], Description):
            example = {"number": str(number).strip(),
                       "name": str(name).strip(),
                       "component": str(component).strip(),
                       "description": str(description).strip()}
            corpus.append(example)
    
    with open("data/data.json", "w", encoding="utf-8") as f:
        for example in corpus:
            f.write("{}\n".format(json.dumps(example, ensure_ascii=False)))

