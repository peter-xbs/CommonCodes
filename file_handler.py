# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 6:48 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""
# -*- coding:utf-8 _*-

import json
import os

# import jieba

# __all__ = ["stat_sentence", "dic2items", "seg_sentence", "stopwordslist", "show_correlation_matrix",
#            "chinese_to_arabic", "calc_time", "seg2sens", "sort_dict", "dirlist"]


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        text = f.read()
        # text = text.split("\n")
    return text


def read_txt_lines(file_path):
    content = read_txt(file_path)
    lines = content.split("\n")
    for line_id in range(len(lines)):
        lines[line_id] = lines[line_id].strip()
    return lines


def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        json_object = json.load(f)
    return json_object


def save_json(file_path, json_object):
    test_and_mkdirs(file_path)
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(json_object, f, ensure_ascii=False, indent=2)


def save_txt(file_path, lines):
    test_and_mkdirs(file_path)
    if isinstance(lines, list):
        lines = "\n".join(lines)
    elif isinstance(lines, set):
        lines = "\n".join(lines)
    else:
        lines = str(lines)
    with open(file_path, 'w', encoding='utf8') as f:
        f.writelines(lines)

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

def copy_file(src_file, dst_file):
    """此函数的功以实现复制文件
    src_file : 源文件名
    dst_file : 目标文件名
    """
    try:
        fr = open(src_file, "rb")  # fr读文件
        try:
            try:
                fw = open(dst_file, 'wb')  # fw写文件
                try:
                    while True:
                        data = fr.read(4096)
                        if not data:
                            break
                        fw.write(data)
                except:
                    print("可能U盘被拔出...")
                finally:
                    fw.close()  # 关闭写文件
            except OSError:
                print("打开写文件失败")
                return False
        finally:
            fr.close()  # 关闭读文件
    except OSError:
        print("打开读文件失败")
        return False
    return True


def read_json_list(file_path):
    from tqdm import tqdm

    in_f = open(file_path, "r", encoding="utf8")
    count = 0
    while 1:
        buffer = in_f.read(8 * 1024 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    in_f.close()

    in_f = open(file_path, "r", encoding="utf8")
    json_list = []

    for line in tqdm(in_f, total=count):
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        try:
            line = json.loads(line)
        except:
            line = eval(line)
        json_list.append(line)
    return json_list


def xml_reader(file_path):
    import xlrd
    workbook = xlrd.open_workbook(file_path, on_demand=True)  # 打开文件
    sheet_name = workbook.sheet_names()  # 所有sheet的名字
    sheets = workbook.sheets()  # 返回可迭代的sheets对象
    result_dict = {}
    for i, sheet in enumerate(sheets):
        result_lines = []
        nrows = workbook.sheet_by_index(i).nrows
        ncols = workbook.sheet_by_index(i).ncols
        for p in range(nrows):
            line = {}
            for q in range(ncols):
                try:
                    str_value = workbook.sheet_by_index(i).cell_value(p, q)
                    line[q] = str_value
                except:
                    pass
            result_lines.append(line)
        result_dict[sheet.name] = result_lines
    return result_dict


def csv_reader(filepath, sep=","):
    import pandas as pd

    # 读取整个csv文件
    csv_data = pd.read_csv(filepath, sep=sep)
    return csv_data.to_dict("records")


def read_pkl(file_path):
    import pickle
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    result = pickle.loads(bytes_in)
    return result


def save_pkl(file_path, json_object):
    test_and_mkdirs(file_path)
    import pickle
    max_bytes = 2 ** 31 - 1
    data = json_object
    bytes_out = pickle.dumps(data)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def test_and_mkdirs(file_path):
    dir_path, file_name = os.path.split(file_path)
    if not os.path.exists(dir_path) and len(dir_path) != 0:
        os.makedirs(dir_path)


def merge_pdf_files(files, tgt_pdf):
    from PyPDF2 import PdfFileMerger
    merger = PdfFileMerger()
    for pdf in files:  # 从所有文件中选出pdf文件合并
        merger.append(pdf)

    merger.write(tgt_pdf)

