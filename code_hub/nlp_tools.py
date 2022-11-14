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