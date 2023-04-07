# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/24 8:24 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""
import copy
import requests


def query_ner(content, type_='XBS'):
    temp = {
        "args": {
            "input_json": {
                "content": "反复咳嗽5天，发热咽喉疼痛2天。",
                "type": "XBS"
            }
        }
    }
    url = 'http://118.31.250.16/struct/api/algorithm/emr_struct'

    req = copy.deepcopy(temp)
    if content.strip():
        req['args']['input_json']['content'] = content
    if type_:
        req['args']['input_json']['type'] = type_

    res = requests.post(url, json=req).json()
    return res


def query_sym_norm(text):
    url = 'http://118.31.52.153:80/api/algorithm/std_norm_api'
    tmpl = {
        "args": {
            "query": "",
            "type": "sym"
        }
    }
    tmpl['args']['query'] = text
    rsp = requests.get(url, json=tmpl).json()
    res = rsp['data']['result']['results']
    if not res:
        return []
    else:
        norm = []
        for sdic in res:
            print()
            norm.append(sdic['norm_res'])
        return norm


def google_translation(queries, dest="zh-CN"):
    """
    注意:  pip install googletrans==3.1.0a0
    REF: https://py-googletrans.readthedocs.io/en/latest/
    调用Google翻译 API
    :param query:
    :param dest:
    :return:
    """
    from googletrans import Translator
    trans = Translator()
    dic = {}
    res = trans.translate(queries, dest=dest)
    for trans_res in res:
        k, v = trans_res.origin, trans_res.text
        dic[k] = v
    return dic

def download_huggingface_dataset(dataset_name, private=False):
    """
    许多用法可参考https://blog.csdn.net/qq_56591814/article/details/120653752
    """
    from datasets import load_dataset
    if private == True:
        dataset_dict = load_dataset(
            dataset_name, # huggingface上对应的name
            use_auth_token='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')
    else:
        dataset_dict = load_dataset(
            dataset_name
        )
    # 从dataset_dict中获取train/test等具体dataset
    dataset = dataset_dict['train'] # 此时Object为Dataset类型
    # dataset.to_csv('保存本地') # 类似to_json(), to_parquet()
    # 对应load_dataset('parquet', data_files={'train': 'xx.parquet'})
    # 或者遍历筛选
    # 或者整体保存至disk dataset.save_to_disc('xx.dataset')
    # 加载 dataset = load_from_disk('xx.dataset')
    # 使用时具体可参考文档

def language_classify(text):
    """
    检测文本语言归属
    """
    # !pip install langid
    import langid
    return langid.classify(text)

def encoding_detect(inp):
    """
    检测文件编码
    """
    import chardet
    with open(inp, 'rb') as f:
        s = f.read()
        res = chardet.detect(s)
        encoding = res.get('encoding')
        return encoding

if __name__ == '__main__':
    pass