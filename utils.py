# -*- encoding: utf-8 -*-
'''
@File   :   utils.py
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   公共函数
'''
import json


def load_data(in_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data