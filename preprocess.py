# -*- encoding: utf-8 -*-
'''
@File   :   preprocess.py
@Time   :   2021/04/26 14:22:39
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   None
'''

import re
from tqdm import tqdm
import xlwt
import random
from utils import load_data


def process(data, with_product=False, with_description=False, with_fine_grained=False):
    """
    1. 仅仅是故障摘要(diagose abstract)数据
    2. 带故障部件数据
    3. 带故障描述数据
    4. 带故障细粒度描述数据
    """
    processed_data = []
    for sample in tqdm(data):
        number = sample["number"]
        name = sample["name"]
        component = sample["component"]
        description = sample["description"]
        
        # 从name字段中抽取product和abstract
        if "[" in name and "]" in name and ":" not in name:
            product = "".join(re.findall(r"\[[^\[\]]*\]", name))
            abstract = name.replace(product, "")
        elif "[" not in name and "]" not in name and ":" not in name:
            product = ""
            abstract = name
        elif "[" in name and "]" in name and ":" in name:
            index = name.find(":")
            if index <= 30:
                product = name[:index]
                abstract = name[index+1:]
            else:
                if "[" in name and "]" in name:
                    product = "".join(re.findall(r"\[[^\[\]]*\]", name))
                    abstract = name.replace(product, "")
                else:
                    product = ""
                    abstract = name
        if product and product[-1] == ":":
            product = product[:-1]
        if abstract and abstract[0] == ":":
            abstract = abstract[1:]
        
        # 从description字段中抽取出细粒度的description
        descrips = description.split("\n")
        recreation_procedure = []
        problem_isolation = []
        for ele in descrips:
            if re.findall(r"\d+\.[^\d]", ele):
                recreation_procedure.append(ele.strip())
            if "(" not in ele and re.findall(r"[A-Z]\)", ele):
                problem_isolation.append(ele.strip())
        
        recreation_procedure = " ".join(recreation_procedure)
        problem_isolation = " ".join(problem_isolation)

        element = {"number": number,
                   "product": product,
                   "component": component,
                   "abstract": abstract,
                   "description": description,
                   "recreation_procedure": recreation_procedure,
                   "problem_isolation": problem_isolation}
        processed_data.append(element)
    return processed_data


if __name__ == "__main__":
    data = load_data("data/data.json")
    processed_data = process(data)

    # 将将处理后的数据全部存入json文件中
    with open("data/processed_data.json", "w", encoding="utf-8") as f:
        for example in processed_data:
            f.write("{}\n".format(json.dumps(example, ensure_ascii=False)))

    # 将处理后的数据抽样1000条存入excel文件中
    random.shuffle(processed_data)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet")
    sheet.write(0, 0, "Number")
    sheet.write(0, 1, "Product")
    sheet.write(0, 2, "Component")
    sheet.write(0, 3, "Abstract")
    sheet.write(0, 4, "Description")
    sheet.write(0, 5, "Recreation Procedure")
    sheet.write(0, 6, "Problem Isolation")

    for idx, sample in enumerate(processed_data[:1000]):
        sheet.write(idx+1, 0, sample["number"])
        sheet.write(idx+1, 1, sample["product"])
        sheet.write(idx+1, 2, sample["component"])
        sheet.write(idx+1, 3, sample["abstract"])
        sheet.write(idx+1, 4, sample["description"])
        sheet.write(idx+1, 5, sample["recreation_procedure"])
        sheet.write(idx+1, 6, sample["problem_isolation"])
    workbook.save("data/processed_data.xlsx")

