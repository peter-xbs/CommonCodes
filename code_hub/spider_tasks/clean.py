# _*_ coding:utf-8 _*_
# @Time: 2023-07-24 11:16
# @Author: cmcc
# @Email: xinbao.sun@hotmail.com
# @File: clean.py
# @Project: CommonCodes

"""
数据清洗方案 主推 readability-lxml + html2text + lxml.html + BeautifulSoup四元组合，完成各类清洗工作
"""
import re
import os
import copy
import traceback
import time
import json
import random
import requests
import pandas as pd
import numpy as np
from selenium import webdriver
from multiprocessing.dummy import Pool as tp
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup as BS

import lxml.html
import html2text
html_txt = html2text.HTML2Text()
html_txt.ignore_links = True
html_txt.ignore_images = True
html_txt.ignore_tables = True
from readability import Document

import glob
inps = glob.glob('/mnt/data3/apps/LLM-prompt-data/DATA/国资委/**/*htm*.html', recursive=True)

with open("/mnt/data3/apps/LLM-prompt-data/DATA/国资委/all_except_主站.jsonl", 'w') as fo:
    for inp in inps:
        if 'png' in inp or '主站' in inp:
            continue
        with open(inp) as f:
            html = f.read()
            doc = Document(html)
            title = doc.title()
            content = doc.summary(html_partial=True)
            content = html_txt.handle(content)
            title = html_txt.handle(title)
#             print('title:', title)
#             print('content:', content)
            cur = {"file": inp, 'title': title, 'content': content}
            line = json.dumps(cur, ensure_ascii=False) + '\n'
            fo.write(line)