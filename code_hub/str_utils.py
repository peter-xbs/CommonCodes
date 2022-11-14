# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 6:47 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""

import copy
import os
import re
from collections import Counter
from functools import reduce


def is_over(index1, index2):
    '''
    判断两个句子的交叠情况
    :param index1:
    :param index2:
    :return:
    '''
    assert len(index1) == 2 and len(index2) == 2
    if index1[0] >= index2[1]:
        return 0

    if index2[0] >= index1[1]:
        return 0

    if index1[0] == index2[0] and index1[1] == index2[1]:
        return 5

    if index1[0] >= index2[0] and index1[1] <= index2[1]:
        return 1

    if index2[0] >= index1[0] and index2[1] <= index1[1]:
        return 2

    if index1[0] <= index2[0] and index1[1] >= index2[0]:
        return 3

    if index2[0] <= index1[0] and index2[1] >= index1[0]:
        return 4


def load_stopwords(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def remove_punctuation(line):
    punctuation = """!"#$%&'()*+,-./:;<=>?@\\[^]_`{|}~！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    new_line = ""
    for c in line:
        if c in punctuation:
            new_line += " "
        else:
            new_line += c
    return new_line



def match_punctuation(start_index, index_offset, sen_indexs, his_text):
    punc_stack = []
    punc_mapper = {
        "(": ")",
        "[": "]",
        "【": "】",
        "{": "}",
        "（": "）",
        "“": "”",
    }
    rever_punc_mapper = dict(zip(punc_mapper.values(), punc_mapper.keys()))
    sen_index = sen_indexs[start_index]
    in_text = his_text[sen_index[0]:sen_index[1]]
    # for char in in_text:
    #     if char in punc_mapper.keys():
    #         punc_stack.append(char)
    #         break
    end_index = start_index

    last_index = start_index + index_offset if start_index + index_offset < len(sen_indexs) else len(
        sen_indexs)
    for index in range(start_index, last_index):
        sen_index = sen_indexs[index]
        in_text = his_text[sen_index[0]:sen_index[1]]
        for char in in_text:
            if char in punc_mapper.values():
                if len(punc_stack) == 0:
                    break
                top_char = punc_stack.pop(-1)
                if top_char == rever_punc_mapper[char]:
                    continue
                else:  # 病例书写错误，直接返回
                    punc_stack = []
                    break
            elif char in punc_mapper.keys():
                punc_stack.append(char)
            elif char in ['"']:
                if len(punc_stack) != 0 and punc_stack[-1] == char:
                    punc_stack.pop(-1)
                    continue
                else:
                    punc_stack.append(char)
        if len(punc_stack) == 0:
            end_index = index
            break
    if len(punc_stack) != 0:
        end_index = start_index
    return end_index



def _seg2indexs(sentence: str, seg_pattern: str) -> list:
    match_results = re.finditer(seg_pattern, sentence)
    index_buffer_list = []
    for match_result in match_results:
        start, end = match_result.span(0)
        index_buffer_list.append([start, end])

    sens = [0]
    for index, item in enumerate(index_buffer_list):
        sens.append(item[0])
        sens.append(item[1])
    sens.append(len(sentence))
    sens = [[sens[index], sens[index + 1]] for index in range(0, len(sens), 2)]

    result_sens = []
    for sen in sens:
        if sen[0] == sen[1]:
            continue
        else:
            result_sens.append(sen)
    return result_sens


def seg2indexs(sentence: str) -> list:
    """
    将句子分割成单独的句子
    :param sentence:
    :return:
    """
    seg_pattern = "([,?!;，。？！；\n\r]+)"
    sub_sen_pattern = "([.: ： ])+"
    num_char = "一二三四五六七八九十两半几多数"
    sens = _seg2indexs(sentence, seg_pattern)

    result_sens = []
    for sen in sens:
        start, end = sen
        if "." in sentence[start:end] or ":" in sentence[start:end] or "：" in sentence[start:end] or ' ' in sentence[
                                                                                                            start:end]:
            sub_sen_text = sentence[start:end]
            sub_sens = _seg2indexs(sub_sen_text, sub_sen_pattern)

            if len(sub_sens) < 1:
                continue

            temp_result_sens = [sub_sens[0]]
            cur_sen_index = temp_result_sens[-1]
            for idx in range(1, len(sub_sens)):
                char_pre = sub_sen_text[sub_sens[idx - 1][1] - 1]
                char_aft = sub_sen_text[sub_sens[idx][0]]

                if is_chinese(char_pre) and is_chinese(
                        char_aft) and char_pre not in num_char and char_aft not in num_char:
                    temp_result_sens.append(sub_sens[idx])
                    cur_sen_index = temp_result_sens[-1]
                else:
                    cur_sen_index[1] = sub_sens[idx][1]

            for temp_result_sen in temp_result_sens:
                result_sens.append([temp_result_sen[0] + start, temp_result_sen[1] + start])

        else:
            result_sens.append(sen)
    filter_result = []
    for sen_index in result_sens:
        sen = sentence[sen_index[0]:sen_index[1]]
        if len(sen.strip()) == 0:
            continue
        filter_result.append(sen_index)
    new_sen_indexs = []
    index = 0
    while index < len(filter_result):
        sen_index = list(filter_result[index])

        in_text = sentence[sen_index[0]:sen_index[1]]
        if re.match('.*[\'\"\(\)“”\[\]【】\{\}（）].*', in_text) is None:
            new_sen_indexs.append(sen_index)
            index += 1
        else:

            end_index = match_punctuation(index, 4, filter_result, sentence)
            sen_index[1] = filter_result[end_index][1]
            index = end_index + 1
            new_sen_indexs.append(sen_index)
    sen_indexs = new_sen_indexs

    if len(sen_indexs) > 0:
        sen_indexs.reverse()
        new_sen_indexs = [sen_indexs.pop()]

        while len(sen_indexs) > 0:
            sen_index_last = new_sen_indexs[-1]
            sen_index = sen_indexs.pop()
            sub_sen = sentence[sen_index_last[1]:sen_index[0] + 1]
            if sub_sen.strip() in [",:", "，：", ",：", "，:"]:
                sen_index_last[1] = sen_index[1]
            else:
                new_sen_indexs.append(sen_index)
        return new_sen_indexs

    return sen_indexs

def is_chinese(char):
    pattern_num_comma = r"[\u4E00-\u9FA5]"
    return re.match(pattern_num_comma, char)


def stat_sentence(str_list):
    data = dict(Counter(str_list))
    return data


def show_correlation_matrix(keys, matrix):
    keys = list(keys)
    w, h = matrix.shape
    assert w == h
    assert w == len(keys)
    str_row_elem = ["{0:5}".format(key) for key in keys]
    str_row = reduce(lambda x, y: x + y, str_row_elem)
    print(" " * 5 + str_row)
    for i in range(h):
        str_row_elem = ["{0:4.4} ".format(num) for num in matrix[i, :]]
        str_row = reduce(lambda x, y: x + y, str_row_elem)
        str_row = "{0:5}".format(keys[i]) + ": " + str_row
        print(str_row)


common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
common_used_numerals = {}
for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]


def chinese2digits(uchars_chinese):
    total = 0
    r = 1  # 表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = common_used_numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                # total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total


num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',
                        '十']
more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']


def chinese_to_arabic(oriStr):
    lenStr = len(oriStr)
    aProStr = ''
    if lenStr == 0:
        return aProStr

    hasNumStart = False
    numberStr = ''
    for idx in range(lenStr):
        if oriStr[idx] in num_str_start_symbol:
            if not hasNumStart:
                hasNumStart = True

            numberStr += oriStr[idx]
        else:
            if hasNumStart:
                if oriStr[idx] in more_num_str_symbol:
                    numberStr += oriStr[idx]
                    continue
                else:
                    numResult = str(chinese2digits(numberStr))
                    numberStr = ''
                    hasNumStart = False
                    aProStr += numResult

                aProStr += oriStr[idx]
                pass

            if len(numberStr) > 0:
                resultNum = chinese2digits(numberStr)
                aProStr += str(resultNum)

            return aProStr

def doc_2_sens(text, max_len, is_overlap=False):
    text = "".join(text)
    sen_indexs = seg2indexs(text)
    result_sens = []
    sen = []
    for index, sen_index in enumerate(sen_indexs):
        sen_index = list(sen_index)

        if index == len(sen_indexs) - 1:
            sen_index[1] = len(text)
        else:
            sen_index[1] = sen_indexs[index + 1][0]

        cur_sen_len = sen_index[1] - sen_index[0]
        if len(sen) > 0:
            sen_len = sen[1] - sen[0]
            if sen_len + cur_sen_len > max_len:  # 如果句子过长，那么把之前的句子先加进段落中
                result_sens.append(sen)
                sen = sen_index
            else:  # 否则更新句子长度
                sen[1] = sen_index[1]

            # 如果句子进入结尾，那么把句子加入到段落中
            if text[sen[1] - 1] in ".。？！?!":
                sen[1] = sen_index[1]
                result_sens.append(sen)
                sen = []
        else:
            sen = sen_index
    if len(sen) != 0:
        result_sens.append(sen)

    if is_overlap:
        new_result_sens = []
        for result_sen in result_sens:
            start, end = result_sen
            if end - start > max_len:
                cur_start = start
                cur_end = cur_start + max_len
                new_result_sens.append([cur_start, cur_end])
                while True:
                    cur_start = cur_end - 20
                    cur_end = cur_start + max_len
                    if cur_end > end:
                        cur_end = end
                        new_result_sens.append([cur_start, cur_end])
                        break
                    else:
                        new_result_sens.append([cur_start, cur_end])
            else:
                new_result_sens.append(result_sen)

        return new_result_sens
    return result_sens


def dirlist(path, allfile, suffix=None, prefix=None):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile, suffix, prefix)
        else:
            suffix_fit = True
            prefix_fit = True
            filename = os.path.split(filepath)[-1]
            if suffix is not None:
                if not filename.endswith(suffix):
                    suffix_fit = False
            if prefix is not None:
                if not filename.startswith(prefix):
                    prefix_fit = False
            if suffix_fit and prefix_fit:
                allfile.append(filepath)
def word_list_n_gram(words, n_gram=1):
    assert n_gram >= 1
    results = []
    for word_index in range(len(words)):
        # results.append(words[word_index])
        for offset in range(n_gram):
            if offset + word_index < len(words):
                sen = "".join(list(words[word_index: word_index + offset + 1]))
                results.append(sen)
    return results


def PowerSetsRecursive(items):
    """Use recursive call to return all subsets of items, include empty set"""

    if len(items) == 0:
        # if the lsit is empty, return the empty list
        return [[]]

    subsets = []
    first_elt = items[0]  # first element
    rest_list = items[1:]

    # Strategy:Get all subsets of rest_list; for each of those subsets, a full subset list
    # will contain both the original subset as well as a version of the sebset that contains the first_elt

    for partial_sebset in PowerSetsRecursive(rest_list):
        subsets.append(partial_sebset)
        next_subset = partial_sebset[:] + [first_elt]
        subsets.append(next_subset)
    return subsets


def PowerSetsRecursive2(items):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    result = sorted(result, key=lambda k: len(k), reverse=True)
    return result


def PowerSetsRecursive_limit(items, max_elem):
    from copy import deepcopy
    result_set = [set()]

    items = list(range(len(items)))
    for _ in range(max_elem):
        new_result_set = []
        for sub_set in result_set:
            for item in items:
                sub_set = deepcopy(sub_set)
                if item not in sub_set:
                    sub_set.add(item)
                    new_result_set.append(sub_set)
        result_set.extend(new_result_set)
    return result_set


def sort_dict(input_dict):
    input_dict = copy.deepcopy(input_dict)
    input_dict = zip(input_dict.keys(), input_dict.values())
    input_dict = sorted(input_dict, key=lambda x: x[1], reverse=True)
    return input_dict


def del_not_chinese(sen):
    pattern_num_comma = r"[^\u4E00-\u9FA5]+"
    return re.sub(pattern_num_comma, "", sen)


def md5(s: str):
    from hashlib import md5 as ori_md5
    return ori_md5(s.encode("utf8")).hexdigest()
def orderd_sim(a, b):
    '''
    计算两个字符串的最长匹配字符串，并返回匹配字数，具体匹配字的序号
    :param a:
    :param b:
    :return:
    '''
    lena = len(a)
    lenb = len(b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]

    pp = []
    for i in range(lena + 1):
        p = []
        for j in range(lenb + 1):
            p.append(["", ""])
        pp.append(p)
    pp[0][0][0] = '#'
    pp[0][0][1] = '#'

    for i in range(1, lena + 1):
        pp[i][0][0] = a[0:i]
        pp[i][0][1] = '#' * len(a[0:i])

    for i in range(1, lenb + 1):
        pp[0][i][0] = '#' * len(b[0:i])
        pp[0][i][1] = b[0:i]

    for i in range(lena):
        for j in range(lenb):
            if a[i] == b[j]:

                c[i + 1][j + 1] = c[i][j] + 1
                pp[i + 1][j + 1][0] = pp[i][j][0] + a[i]
                pp[i + 1][j + 1][1] = pp[i][j][1] + b[j]

            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
                pp[i + 1][j + 1][0] = pp[i + 1][j][0] + '#'
                pp[i + 1][j + 1][1] = pp[i + 1][j][1] + b[j]

            else:
                c[i + 1][j + 1] = c[i][j + 1]
                pp[i + 1][j + 1][0] = pp[i][j + 1][0] + a[i]
                pp[i + 1][j + 1][1] = pp[i][j + 1][1] + '#'

    out_stra = pp[lena][lenb][0]
    out_strb = pp[lena][lenb][1]

    assert len(out_stra) == len(out_strb)

    match_char_index_a = []
    match_char_index_b = []
    char_index_a = 0
    char_index_b = 0
    for char_index in range(len(out_stra)):
        if out_strb[char_index] == out_stra[char_index] and out_stra[char_index] != '#':
            match_char_index_b.append(char_index_b)
            match_char_index_a.append(char_index_a)
        if out_strb[char_index] != '#':
            char_index_b += 1
        if out_stra[char_index] != '#':
            char_index_a += 1

    return c[lena][lenb], match_char_index_a, match_char_index_b

def remove_duplication_index(ys_indexs: list, used_indexs):
    # TODO 可以对迭代算法进行优化
    ys_indexs = sorted(ys_indexs, key=lambda x: x[0][0])
    used_indexs = sorted(used_indexs, key=lambda x: x[0][0])

    if len(used_indexs) == 0:
        used_indexs = ys_indexs
        return used_indexs

    append_indexs = []

    for ys_index in ys_indexs:
        ys_index, ys_tag = ys_index
        is_duplicated = False
        for used_index in used_indexs:
            used_index, _ = used_index
            if is_over(ys_index, used_index) != 0:
                is_duplicated = True
                break
        if not is_duplicated:
            append_indexs.append([ys_index, ys_tag])

    [used_indexs.append(ys_index_pair) for ys_index_pair in append_indexs]
    return used_indexs


def normal_str(old_str):
    import unicodedata
    new_str = []
    for char in old_str:
        new_char = unicodedata.normalize("NFKC", char)
        if len(new_char) != len(char):
            new_char = " "
        if 0 <= ord(char) <= 31 or ord(char) == 127:
            new_char = " "
        if char in ['℃']:
            new_char = char

        new_str.append(new_char)
    new_str = "".join(new_str)
    new_str = new_str.lower()
    return new_str


def normal_quote(old_str: str):
    e_pun = u'￥,!?[]()<>"\'～'
    c_pun = u'$，！？【】（）《》“‘~'
    table = {ord(f): ord(t) for f, t in zip(c_pun, e_pun)}
    return old_str.translate(table)




def remove_duplicate(item_list):
    item_set = set()
    result_item_list = []
    for item in item_list:
        item_str = str(item)
        if item_str not in item_set:
            item_set.add(item_str)
            result_item_list.append(item)
    return result_item_list


def sep_long_sentence(content, max_len):
    temp_sen_indexs = seg2indexs(content)
    if len(temp_sen_indexs) == 0:
        return []
    sen_indexs = []
    start_index = temp_sen_indexs[0][0]
    end_index = temp_sen_indexs[0][1]
    for sen_index in temp_sen_indexs:
        if sen_index[1] - start_index > max_len:
            sen_indexs.append([start_index, end_index])
            start_index = sen_index[0]
            end_index = sen_index[1]
        else:
            end_index = sen_index[1]
        sen_indexs.append([start_index, len(content)])
        sen_indexs = remove_duplicate(sen_indexs)
        return sen_indexs

    def detect_encoding(file_path):
        import chardet
        with open(file_path, 'rb') as f:
            s = f.read()
            res = chardet.detect(s)
            encoding = res.get('encoding')
        return encoding

    def strQ2B(ustring):
        """
        全角转半角
        Args:
            ustring:

        Returns:

        """
        if isinstance(ustring, str):
            rstring = ""
            for uchar in ustring:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248

                rstring += chr(inside_code)
            return rstring
        return ustring

    if __name__ == '__main__':
        pass