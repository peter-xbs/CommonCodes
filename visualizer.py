# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 6:43 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""
color_schema = {
    "red": ["Salmon", "LightSalmon", "Crimson", "Red", "FireBrick"],
    "pink": ["Pink", "HotPink", "DeepPink", "MediumVioletRed"],
    "orange": ["Coral", "OrangeRed", "Orange"],
    "yellow": ["Gold", "Yellow", "Moccasin", "PaleGoldenrod", "DarkKhaki"],
    "purple": ["Orchid", "Fuchsia", "MediumPurple", "RebeccaPurple", "BlueViolet", "Purple", "Indigo", "SlateBlue"],
    "green": ["GreenYellow", "Lime", "LimeGreen", "SeaGreen", "Green", "YellowGreen", "Olive", "LightSeaGreen"],
    "blue": ["Cyan", "LightCyan", "SteelBlue", "LightSteelBlue", "LightSkyBlue", "DeepSkyBlue", "DodgerBlue",
             "CornflowerBlue", "MediumSlateBlue", "RoyalBlue"],
    "brown": ["NavajoWhite", "Tan", "RosyBrown", "SandyBrown", "Goldenrod", "Chocolate", "Sienna", "Brown"],
    "white": ["HoneyDew", "Azure", "GhostWhite", "Beige", "LavenderBlush"],
    "grey": ["Silver", "Gray", "LightSlateGray", "DarkSlateGray"],
}

supply_color_db = [
    "LightCoral", "DarkRed", "IndianRed", "DarkSalmon", "PaleVioletRed", "LightPink", "Tomato", "DarkOrange",
    "LightSalmon", "Khaki",
    "PapayaWhip", "LemonChiffon", "PeachPuff", "LightGoldenrodYellow", "LightYellow", "Plum", "MediumSlateBlue",
    "DarkViolet", "Thistle",
    "DarkMagenta", "MediumOrchid", "DarkSlateBlue", "Lavender", "Violet", "Magenta", "DarkOrchid", "MediumSeaGreen",
    "Chartreuse", "Teal",
    "MediumAquamarine", "OliveDrab", "LightGreen", "DarkCyan", "LawnGreen", "DarkSeaGreen", "PaleGreen", "ForestGreen",
    "DarkGreen",
    "MediumSpringGreen", "DarkOliveGreen", "SpringGreen", "Navy", "Aquamarine", "DarkTurquoise", "DarkBlue", "Aqua",
    "PowderBlue",
    "MidnightBlue", "MediumBlue", "LightBlue", "SkyBlue", "Blue", "Turquoise", "CadetBlue", "MediumTurquoise",
    "PaleTurquoise", "Bisque",
    "DarkGoldenrod", "Wheat", "Peru", "BurlyWood", "Cornsilk", "SaddleBrown", "BlanchedAlmond", "Maroon",
    "AntiqueWhite", "Snow", "WhiteSmoke",
    "MintCream", "SeaShell", "White", "FloralWhite", "OldLace", "AliceBlue", "Ivory", "Linen", "MistyRose", "LightGray",
    "Black", "Gainsboro",
    "DimGray", "DarkGray", "SlateGray"]


def labels2colors(labels):
    """最多支持143种配色"""
    import random
    map_dict = {}
    length = len(labels)
    assert length <= 143
    # less_priority = ['red', 'white', 'brown', 'white', 'gray']
    if length == 1:
        res = ["Cyan"]
    elif length == 2:
        res = ["Cyan", "Orange"]
    else:
        if length <= 10:
            res = []
            for key in color_schema:
                res.append(random.sample(color_schema[key], 1))
        elif length <= 20:
            res = []
            for key in color_schema:
                res.extend(random.sample(color_schema[key], 2))
        elif length <= 30:
            res = []
            for key in color_schema:
                res.extend(random.sample(color_schema[key], 3))
        elif length <= 60:
            all_colors = []
            for key in color_schema:
                all_colors.extend(color_schema[key])
            res = random.sample(all_colors, length)
        else:
            all_colors = []
            for key in color_schema:
                all_colors.extend(color_schema[key])
            all_colors.extend(supply_color_db)
            res = random.sample(all_colors, length)
    random.shuffle(res)
    colors = res[:length]
    for label, color in zip(labels, colors):
        map_dict[label] = color
    return map_dict

def render_med_ner_html(labels, sen):
    '''
    可视化句子标签效果，使用不同的背景颜色表示不同的标签
    :param labels:[[s,e],[l1##,l2##,l3##]]
    :param sen:
    :return:
    '''
    color_dict = {

        "KSSJ": "rgba(199,21,133, 0.5)",
        "CXSJ": "rgba(218,112,214, 0.5)",
        "TIZ": "rgba(127, 255, 0, 0.5)",
        "PL": "rgba(139,0,139, 0.5)",
        "JCJG": "rgba(0,0,15, 0.5)",
        "DIGIT": "rgba(0,0,155, 0.5)",
        "UNIT": "rgba(25,25,0, 0.5)",
        "LCTS": "rgba(0,0,255, 0.5)",
        "YYPL": "rgba(25,25,212, 0.5)",
        "SRFF": "rgba(25,25,212, 0.5)",

        "ZZ": "rgba(220,20,60, 0.5)",
        "YXZZ": "rgba(255,240,24, 0.5)",

        "BW": "rgba(255,182,193)",
        "JC": "rgba(25, 21, 21, 0.5)",
        "JY": "rgba(25, 210, 21, 0.5)",
        "JB": "rgba(25, 210, 201, 0.5)",
        "SS": "rgba(0, 21, 225, 0.5)",
        "YW": "rgba(25, 21, 100, 0.5)",

        "ZLS": "rgba(199,21,133, 0.5)",
        "ZDS": "rgba(218,112,214, 0.5)",
        "YXJC": "rgba(127, 255, 0, 0.5)",
        "XDTJC": "rgba(139,0,139, 0.5)",
        "SYSJC": "rgba(0,0,155, 0.5)",
        "BLJC": "rgba(25,25,0, 0.5)",
        "NJJC": "rgba(0,0,255, 0.5)",
        "RYQK": "rgba(25,25,212, 0.5)",
        "YB": "rgba(25,25,212, 0.5)",

    }
    html_template = r'<span style="background-color:%s">%s</span>'
    htmls = []
    tags = ["NULL"] * len(sen)
    for index, label_item in enumerate(labels):
        label_index = label_item[0]
        tag = label_item[1]
        for i in range(label_index[0], label_index[1]):
            tags[i] = tag

    start = 0
    pre_tag = tags[0]
    for tag_index, tag in enumerate(tags):
        # 当前句子如果有标签的话，打上相应的背景颜色
        if tag == pre_tag:
            continue
        else:
            sub_sen = sen[start:tag_index]
            if pre_tag in color_dict:
                # TODO 同一个句子可能有不同的标签，需要想一下这种情况如何处理
                html_p = html_template % (color_dict[pre_tag], sub_sen)
            else:
                html_p = sub_sen
            start = tag_index
            pre_tag = tag
        htmls.append(html_p)
    sub_sen = sen[start:len(sen)]
    tag = tags[start]
    if tag in color_dict:
        # TODO 同一个句子可能有不同的标签，需要想一下这种情况如何处理
        html_p = html_template % (color_dict[tag], sub_sen)

    else:
        html_p = sub_sen
    htmls.append(html_p)

    htmls.append("<br/>")
    html = "".join(htmls)
    # html = "&nbsp&nbsp".join(htmls)
    return html


def plot_attention(query1, query2, d):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.ticker as ticker

    assert isinstance(query1, list)
    assert isinstance(query2, list)
    d = d.transpose()
    col = [t for t in query2]  # 需要显示的词
    index = [t for t in query1]  # 需要显示的词
    df = pd.DataFrame(d, columns=col, index=index)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # fontdict = {'rotation': 'vertical'}    #设置文字旋转
    fontdict = {'rotation': 90}  # 或者这样设置文字旋转
    # ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
    # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
    ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
    ax.set_yticklabels([''] + list(df.index))

    plt.show()


def plot_embedding_with_tsne(embeddings, tag_list, keep_idx=None, texts=None):
    """
    :param embeddings: 节点的embedding表示 n*m
    :param tag_list: 节点类型 n*1
    :param texts: 节点描述字符串 n*1 不需要添加的字符串的节点id用None表示
    :return:
    """

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embeddings)
    if texts is not None:
        for id in range(len(texts)):
            text = texts[id]
            if text is None:
                continue
            x = node_pos[id][0]
            y = node_pos[id][1]
            plt.text(x, y, text, fontsize="xx-small", fontproperties='SimHei')

    color_idx = {}
    for i in range(len(tag_list)):
        color_idx.setdefault(tag_list[i], [])
        color_idx[tag_list[i]].append(i)

    for c, idx in color_idx.items():
        n_idx = []
        if keep_idx is not None:
            for id in idx:
                if id in keep_idx:
                    n_idx.append(id)
            else:
                n_idx = idx
            plt.scatter(node_pos[n_idx, 0], node_pos[n_idx, 1], label=c)
            plt.legend()
            plt.show()

        def plot_word_cloud(file_path, words):
            # 网页版 提供一种替代
            # REF: https://www.jasondavies.com/wordcloud/
            from wordcloud import WordCloud
            # 设置字体，不指定就会出现乱码 # 设置背景色
            # font_path = r'C:\Windows\Fonts\simsun.ttc'
            wc = WordCloud(
                # font_path=font_path,
                background_color='white',  # 设置背景宽
                width=500,  # 设置背景高
                height=350,  # 最大字体
                max_font_size=50,  # 最小字体
                min_font_size=10,
                mode='RGBA',
                max_words=2000,
                colormap='pink',
                collocations=False)  # 产生词云
            result = " ".join(words)
            wc.generate(result)  # 保存图片
            wc.to_file(file_path)  # 按照设置的像素宽高度保存绘制好的词云图，比下面程序显示更清晰

        def plot_word_cloud2(text, stop_words):
            from matplotlib import pyplot as plt
            import wordcloud
            import jieba.posseg as pseg
            # font_path = "/System/Library/Fonts/PingFang.ttc",

            # 去掉所有动词
            words = pseg.cut(text)
            keep_words = []
            for word, flag in words:
                if not flag.startswith('v') and len(word) > 1:
                    keep_words.append(word)
            print(len(keep_words))

            # 去掉一些无用词
            custom_stop_words = stop_words

            wordcloud_notes = wordcloud.WordCloud(
                stopwords=custom_stop_words,
                max_font_size=120,
                max_words=5000,
                width=600,
                height=400,
                background_color='white').generate(" ".join(keep_words))
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(wordcloud_notes, interpolation='bilinear')
            ax.set_axis_off()
            plt.imshow(wordcloud_notes)

        def ner_show(text, out):
            """
            Ref: https://spacy.io/usage/visualizers#html
            # 另一部分参见nlp_api中的visualize_emr_ner
            """
            import spacy
            import jieba.posseg as pseg
    # 构造数据
    words = pseg.cut(text)
    ents = []
    cur_start = 0
    for word, flag in words:
        if flag=='v':
            ents.append({
                'start': cur_start,
                'end': cur_start+len(word),
                'label': 'verb'
            })
        cur_start+=len(word)

    doc = {
        'text' : text,
        "ents" : ents,
        "title": "测试"
    }
    # colors = {"verb" :"linear-gradient(90deg, #aa9cfc, #fc9ce7)" }
    # options = {"colors": colors, "lang": "zh"}
    options = {}

    html = spacy.displacy.render([doc,doc,doc], style="ent", options = options , manual=True, page=True)
    with open(out, 'w') as fo:
        fo.write(html)

def visualize_emr_ner(texts):
    from hub.nlp_tools import query_ner
    import spacy
    def build_doc(rsp, idx, label_dict):
        def is_sub_entity(entity, ents):
            (cur_st, cur_ed), _ = entity
            for tup in ents:
                pos, tpe = tup
                st, ed = pos
                if st == cur_st and ed == cur_ed:
                    continue
                if st <= cur_st and ed >= cur_ed:
                    return True
            return False

        result = rsp['data']['result']
        text = result['content']
        entities = result['entities']
        ents = []
        for tup in entities:
            if is_sub_entity(tup, entities):
                continue
            pos, tpe_ = tup
            st, ed = pos
            tpe = label_dict.get(tpe_)
            if not tpe:
                print(tpe_, text[st:ed])
                continue
            dic = {
                'start': st,
                'end': ed,
                'label': tpe
            }
            ents.append(dic)
        ents.sort(key=lambda x: x['start'])
        doc = {
            'text': text,
            "ents": ents,
            "title": "文本{}".format(idx)
        }
        return doc
    label_dict = {
        'JB': '疾病', 'BLFX': '病理分型', 'BLFJ': '病理分级', 'BLFQ': '病理分期', 'ZYBW': '转移部位',
        'KSSJ': '开始时间', 'CXSJ': '持续时间', 'BW': '部位', 'CD': '程度', 'XINGZ': '性质', 'DX': '大小',
        'XZ': '形状', 'BY': '病原', 'FB': '分布', 'LA': '量', 'SSLX': '手术类型', 'MZFS': '麻醉方式',
        'SS': '手术', 'YW': '药物', 'YYJL': '用药剂量', 'YYPC': '用药频次', 'SRFS': '摄入方式',
        'HLTS': '化疗疗程天数', 'ZLXG': '治疗效果', 'BLFY': '不良反应', 'JY': '检验项目', 'DIGIT': '数值',
        'UNIT': '单位', 'JYYC': '检验异常', 'JYZC': '检验正常', 'JC': '检查', 'ZZ': '症状', 'YXZZ': '阴性症状',
        'EHYS': '恶化因素', 'HJYS': '缓解因素', 'WGYS': '无关因素', 'YFYS': '诱发因素', 'TIZ': '体征', 'BSFMW': '伴随分泌物',
        'ZG': '转归', 'FSPL': '发生频率', 'YS': '颜色', 'CS': '次数', 'QW': '气味', 'PL': '频率', 'YQSJX': '孕期时间项',
        'YCS': '孕产史', 'YJD': '孕阶段', 'TW': '胎位', 'QTZL': '其他治疗', 'YXJB': '否认疾病', 'GMY': '过敏源',
        'GMBX': '过敏表现', 'QTBS': '其它病史', 'FRQTBS': '否认其它病史'}
    docs = []
    for idx, text in enumerate(texts):
        rsp = query_ner(text)
        doc = build_doc(rsp, idx, label_dict)
        docs.append(doc)
    map_dict = labels2colors(list(label_dict.values()))
    options = {"colors": map_dict}
    html = spacy.displacy.render(docs, style="ent", options=options, manual=True, page=True)
    return html


def labels2col_codes(labels):
    def rgb_to_hex(rgb_triplet) -> str:
        """
        ref: python package `webcolors`
        Convert a 3-tuple of integers, suitable for use in an ``rgb()``
        color triplet, to a normalized hexadecimal value for that color.
        """
        rgb_triplet = (0 if value < 0 else 255 if value > 255 else value for value in rgb_triplet)
        return "#{:02x}{:02x}{:02x}".format(*rgb_triplet)
    import random
    import seaborn as sns
    length = len(labels)
    col_tuples = sns.color_palette('Paired', length)
    codes = []
    labels2codes = {}
    for tup in col_tuples:
        tup_ = [int(x*255.0) for x in tup]
        code = rgb_to_hex(tup_)
        codes.append(code)
    random.shuffle(codes)
    for lab, code in zip(labels, codes):
        labels2codes[lab] = code
    return labels2codes


def network_show():
    """
    REF: https://infovis.fh-potsdam.de/tutorials/infovis7networks.html
    """
    pass


def gen_brat_conf(tgt_dir):
    """生成brat的核心配置"""
    import os
    visual_conf = os.path.join(tgt_dir, 'visual.conf')
    anno_conf = os.path.join(tgt_dir, 'annotation.conf')
    tool_conf = os.path.join(tgt_dir, 'tools.conf')
    with open(visual_conf, 'w') as fo:
        keys = ['[labels]', '[drawing]', '[options]']
        fo.write('\n\n'.join(keys))

    with open(anno_conf, 'w') as fo:
        keys = ['[entities]', '[relations]', '[events]', '[attributes]']
        fo.write('\n\n'.join(keys))

    with open(tool_conf, 'w') as fo:
        keys = ['[options]', '[search]', '[annotators]', '[disambiguators]', '[normalization]']
        fo.write('\n\n'.join(keys))

if __name__ == '__main__':
    import seaborn
    text = "国务院总理李克强5日在政府工作报告中提出，大力抓好农业生产，促进乡村全面振兴。完善和强化农业支持政策，接续推进脱贫地区发展，促进农业丰收、农民增收。"
    # ner_show(text, 'test.html')
    fin_set = set()
    label_dict = {
        'JB': '疾病', 'BLFX': '病理分型', 'BLFJ': '病理分级', 'BLFQ': '病理分期', 'ZYBW': '转移部位',
        'KSSJ': '开始时间', 'CXSJ': '持续时间', 'BW': '部位', 'CD': '程度', 'XINGZ': '性质', 'DX': '大小',
        'XZ': '形状', 'BY': '病原', 'FB': '分布', 'LA': '量', 'SSLX': '手术类型', 'MZFS': '麻醉方式',
        'SS': '手术', 'YW': '药物', 'YYJL': '用药剂量', 'YYPC': '用药频次', 'SRFS': '摄入方式',
        'HLTS': '化疗疗程天数', 'ZLXG': '治疗效果', 'BLFY': '不良反应', 'JY': '检验项目', 'DIGIT': '数值',
        'UNIT': '单位', 'JYYC': '检验异常', 'JYZC': '检验正常', 'JC': '检查', 'ZZ': '症状', 'YXZZ': '阴性症状',
        'EHYS': '恶化因素', 'HJYS': '缓解因素', 'WGYS': '无关因素', 'YFYS': '诱发因素', 'TIZ': '体征', 'BSFMW': '伴随分泌物',
        'ZG': '转归', 'FSPL': '发生频率', 'YS': '颜色', 'CS': '次数', 'QW': '气味', 'PL': '频率', 'YQSJX': '孕期时间项',
        'YCS': '孕产史', 'YJD': '孕阶段', 'TW': '胎位', 'QTZL': '其他治疗', 'YXJB': '否认疾病', 'GMY': '过敏源',
        'GMBX': '过敏表现', 'QTBS': '其它病史', 'FRQTBS': '否认其它病史'}

    r = labels2col_codes(list(label_dict.values()))
    for k, v in r.items():
        print(k, 'bgColor:'+v)
    print('#'*60)
    for k in r:
        print(k)