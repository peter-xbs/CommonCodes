# _*_ coding:utf-8 _*_
# @Time: 2023-04-02 20:36
# @Author: cmcc
# @Email: xinbao.sun@hotmail.com
# @File: sent_struct_plot.py
# @Project: CommonCodes

"""
句子依存结构解析后，同时画旭日图：图展示句子结构
"""

import os
import benepar, spacy
import pandas as pd
import re
import plotly.express as px


# 利用benepar分析器对file中pormpt句子做句法分析，结果存入parser.txt文件
def parser_file(file):
    nlp = spacy.load('en_core_web_md')
    # if spacy.__version__.startswith('2'):
    #     nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    # else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    fin = pd.read_csv(file)
    data = fin['prompt'].tolist()
    print('all data is {}'.format(len(data)))
    results = []
    for idx, dat in enumerate(data):
        if idx % 1000 == 0:
            print(idx)
        doc = nlp(str(dat))
        sent = list(doc.sents)[0]
        results.append(sent._.parse_string)

    with open('parser.txt', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(results))
    print('Done')

    # (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
    # print(sent._.labels)
    # ('S',)
    # print(list(sent._.children)[0])
    # The time for action


# 提取单句-陈述句，存入sen_S.txt
def process_parserres():
    sentence_type = {}
    sen_state = []
    with open('parser.txt', 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin.readlines()):
            new_line = line.strip().split()
            if new_line and new_line[0][0] == '(' and new_line[-1][-1] == ')':

                print(idx)
                print(new_line[0][1:])
                if new_line[0][1:] == 'S':
                    sen_state.append(line.strip())

                if new_line[0][1:] not in sentence_type:
                    sentence_type[new_line[0][1:]] = 1
                else:
                    sentence_type[new_line[0][1:]] += 1

    print(sentence_type)
    with open('sen_S.txt', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(sen_state))


# 对句法分析结果构建层级树，仅保留动、名词。
def struct_tree(line):
    i = 0
    line = re.split('(\(|\)| )', line.strip())
    print(line)
    line = [lin for lin in line if lin and lin != ' ']
    print(line)
    new_line = []
    for idx, l in enumerate(line):
        if l == '(':
            i += 1
        if l == ')':
            i -= 1
        if l in ['S', 'VP', 'NP', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']:
            if line[idx + 1] != '(':
                new_line.append(l + '-' + str(i) + '-' + line[idx + 1])
            else:
                new_line.append(l + '-' + str(i))

    # print(new_line)
    return new_line


# line = (S (VP (VB Please) (VP (VB provide) (NP (PRP me)) (PP (IN with) (NP (DT some) (JJ financial) (NN advice))))) (. .))
# print(struct_tree(line))

# 在【'S-1', 'VP-2', 'VB-3-Please', 'VP-3', 'VB-4-provide', 'NP-4', 'NP-5', 'NN-6-advice】中提取句子主干动词、名词
def vn(lis):
    n_list = ['NN', 'NNS', 'NNP', 'NNPS']
    # v_list = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    v_list = ['VB', 'VBG']
    dig_vp = 100

    v_value = ''
    n_value = ''
    for idx, li in enumerate(lis):
        if len(li.split('-')) == 2:
            label, dig = li.split('-')
            if label == 'S':
                vn(lis[idx + 1:])
            if label == 'VP':
                dig_vp = int(dig)
        if len(li.split('-')) == 3:
            label, dig, word = li.split('-')
            if label in v_list and int(dig) > dig_vp:
                v_value = word
            if label in n_list and int(dig) > dig_vp:
                n_value = word
    return v_value, n_value


# 提取句子vp短语中的动词及名词。
# (S (VP (VB Please) (VP (VB provide) (NP (PRP me)) (PP (IN with) (NP (DT some) (JJ financial) (NN advice))))) (. .))
# vp动词短语，
# topK,second_topK分别是选取动词前topK个，对每个动词，选取前second_topK个名词
def extract_vn(topK, second_topK):
    df = [['verb', 'noun', 'num']]
    df_dict = {}
    verb_dict = {}
    n_list = ['NN', 'NNS', 'NNP', 'NNPS']
    v_list = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    with open('sen_S.txt', 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            tree = struct_tree(line)
            # tree: ['S-1', 'VP-2', 'VB-3-Please', 'VP-3', 'VB-4-provide', 'NP-4', 'NP-5', 'NN-6-advice']
            v_value, n_value = vn(tree)
            if v_value:
                if v_value not in df_dict:
                    df_dict[v_value] = {n_value: 1}
                    verb_dict[v_value] = 1
                else:
                    verb_dict[v_value] += 1
                    if n_value:
                        if n_value not in df_dict[v_value]:
                            df_dict[v_value][n_value] = 1
                        else:
                            df_dict[v_value][n_value] += 1

    df1 = [['verb', 'num']]
    verb_tuple = sorted(verb_dict.items(), key=lambda x: x[1], reverse=True)
    verb_topK = []

    for key, value in verb_tuple:
        verb_topK.append(key)
        df1.append([key, value])

    verb_topK = verb_topK[:topK]

    df1 = pd.DataFrame(df1)
    df1.to_csv('verb_data.csv', index=False)

    for key, value in df_dict.items():
        if key in verb_topK:
            n_num_sort = sorted(value.items(), key=lambda x: x[1], reverse=True)
            for k, v in n_num_sort[:second_topK]:
                df.append([key, k, v])
    # 将df写入csv文件
    df = pd.DataFrame(df)
    df.to_csv('sunburstdata.csv', index=False)


# 根据csv文件画旭日图
def plt_pic():
    cur_dir = os.path.dirname(__file__)

    data = pd.read_csv(cur_dir + '/sunburstdata (2).csv')
    # fig=px.sunburst(data,path=['verb','noun'],values='num',color='num',color_continuous_scale='Bugn')
    fig = px.sunburst(data, path=['verb', 'noun'], values='num', color='num')
    fig.update_traces(textinfo='label+value+percent root')
    fig.show()


if __name__ == "__main__":
    extract_vn(20, 4)
