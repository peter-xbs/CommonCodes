# _*_ coding:utf-8 _*_

"""
本模块主要用于大段文本的切割功能
"""

import re


class Seg(object):
    basic_regex1 = '(?<!\d)[：:]'
    basic_regex2 = '\r\n|\n|\r|\s{1,4}|\t{1,2}|。|；|\|\|\|'
    _sent_seg_regex = re.compile('(\r\n|\n|[。；])')
    ext_regex = None
    # 模版切割
    template = None
    freq_seq = None

    @classmethod
    def template_seg(cls, content, freq_seq, mode='relax'):
        """
        使用传入的freq_seq作为模版进行切割
        :param content:
        :param freq_seq:
        :param mode: 包括strict和relax两种，前者要求使用KEY进行切分时,KEY前必须为制表符或者以KEY开始,后者放宽该要求
        :return:
        """
        if cls.template is None:
            cls.load_template(freq_seq, mode)
        if freq_seq != cls.freq_seq:
            cls.load_template(freq_seq, mode)
        # 空模板不作切割
        if not cls.freq_seq:
            return content, []
        splits = cls.template.split(content)
        splits = [item for item in splits if item is not None]

        if not splits:
            return '', []
        length = len(splits)
        nonkey_content = splits[0]
        key_content = []
        for i in range(2, length, 2):
            key = splits[i - 1]
            val = splits[i]
            key_content.append([key, val])
        return nonkey_content, key_content

    @classmethod
    def phy_exm_seg(cls, content, template, sep_flag):
        """
        体格检查专项切割工具
        Args:
                content:
                template:

        Returns:

        """
        # 空模板不作切割
        if not template:
            return content, []
        template.sort(key=len, reverse=True)
        constraint_freq_seq = []
        for item in template:
            item = cls.key_preprocess(item)
            item = '\s*'.join(list(item))
            if sep_flag == 'colon':
                item = "(?:(?<=\s)|(?<=^))" + item + "[:：]"
            elif sep_flag == 'space':
                item = "(?:(?<=\s)|(?<=^))" + item + "\s"
            constraint_freq_seq.append(item)
        template_regex = '(' + '|'.join(constraint_freq_seq) + ')'
        template_regex = re.compile(template_regex)
        splits = template_regex.split(content)
        splits = [item for item in splits if item is not None]
        if not splits:
                return []
        length = len(splits)
        key_content = []
        key_content.append(['non_key_header', splits[0]])
        for i in range(2, length, 2):
                key = splits[i - 1]
                val = splits[i]
                key_content.append([key, val])
        return key_content


    @classmethod
    def repl_trans(cls, m):
            return '\\'.join(list(m.group(0)))

    @classmethod
    def key_preprocess(cls, key):
            """
            为防止特殊key带来的编译错误，提前作转义预处理
            """
            key = key.replace('(', '\(').replace(')', '\)')
            key = re.sub('^[\[*.+]{1,}|\]$', '', key)
            key = re.sub('[.*]{2,}', cls.repl_trans, key)
            return key

    @classmethod
    def load_template(cls, freq_seq, mode):
            """
            加载模版
            :param freq_seq:
            :return:
            """
            freq_list = list(freq_seq)
            freq_list.sort(key=len, reverse=True)
            cls.freq_seq = freq_seq
            constraint_freq_seq = []
            for item in freq_list:
                    item = cls.key_preprocess(item)
                    if mode == 'strict':
                            item_ = '(?:(?<=[\s\n\t])|(?<=^))'+item+'(?![,，])[:：]?'
                    else:
                            item_ = item+'(?![,，])[:：]?'
                    constraint_freq_seq.append(item_)
            template = '(' + '|'.join(constraint_freq_seq) + ')'
            cls.template = re.compile(template)

    @classmethod
    def raw_seg(cls, content, mode='basic', adhesion=False):
            """
            仅根据正则切割文本
            :param content:
            :param mode: colon(:)/space(\s\t)
            :return:
            """
            if not content:
                    return content
            content = content.replace('|||', '  ')
            if mode == 'basic':
                    regex_str = '(' + cls.basic_regex1 + '|' + cls.basic_regex2 + ')'
                    cls.regex = re.compile(regex_str)
            seg_res = []
            splits = cls.regex.split(content)
            length = len(splits)
            for i in range(0, length, 2):
                    sent = splits[i]
                    punct = splits[i+1] if i+1 < length else ''
                    if cls._is_punct(punct):
                            sent_punc = ''.join([sent, punct])
                            if not sent_punc.strip():
                                    continue
                            seg_res.append(sent_punc)
                    else:
                            seg_res.append(sent)
            return seg_res


    @classmethod
    def _is_punct(cls, punct):
        if not punct.strip():
            return True
        if re.match(cls.basic_regex1, punct) or \
                re.match(cls.basic_regex2, punct):
            return True
        return False


    @classmethod
    def split_title_and_content(cls, content):
        contents = re.split('\n|\r\n|\r', content)
        tmp1 = contents[0]
        if len(tmp1) < 40 and ('记录' in tmp1 or '同意书' in tmp1 or '通知单' in tmp1 or '申请书' in tmp1):
            title = tmp1 + '\n'
            content = '\n'.join(contents[1:])
        else:
            title = ''
            content = content
        return title, content


    @classmethod
    def sent_seg(cls, content):
        """
            句子级别切分
            :param content:
            :return:
            """
        seg_res = []
        splits = cls._sent_seg_regex.split(content)
        length = len(splits)
        if length == 1:
            return splits
        for i in range(0, length, 2):
            sent = splits[i]
            punct = splits[i + 1] if i + 1 < length else ''
            if cls._sent_seg_regex.match(punct):
                sent_punc = ''.join([sent, punct])
                if not sent_punc.strip():
                    continue
                seg_res.append(sent_punc)
        return seg_res


if __name__ == '__main__':
    import json

    s = "出院带药： 无 随访计划： 出院 1~ 2周内消化内科复诊查看病理结果，如有不适及时就诊。若病理阴性，建议1年复查胃 肠镜"
    s = "主持 及 参加讨论者姓名职称："
    # print(s)
    # s = " ".join(s.split())
    # print(json.dumps(s, ensure_ascii=False))
    # s2 = "无免疫性和精神性疾病。 体 格 检 查 体温:"
    # sent_list = Seg.raw_seg(s, mode='ext')
    # r = re.split('住院重要检查化验结果：', s)
    # print(r)
    # s = """Subjective: 患者无胸闷气急，无畏寒发热，乏力较前减轻,无恶心呕吐等不适，胃纳及睡眠可，二便无殊。
    # Objective: 神清，精神可，唇不绀，颈静脉无怒张，两肺呼吸音粗，两肺未闻及明显干湿罗音，心界向左扩大，心率90次/分，律不齐，第一心音强弱不等，各瓣膜未闻及明显病理性杂音，腹平软，全腹无压痛，肝脾肋下未及，双下肢无浮肿。四肢肌力5级，肌张力正常，病理征阴性。血常规+CRP 2018-7-5 08:32:57 血红蛋白 92 g/L;C反应蛋白 9.92 mg/L;BNP+cTnI 血清肌钙蛋白 0.046 ng/ml;B型纳尿肽 3800.00 pg/ml;PT 国际标准化比值 3.18 ;小生化+心肌酶谱 总胆红素 28.9 μmol/L;直接胆红素 17.0 μmol/L;尿酸 389.0 μmol/L;肌酐 91.6 μmol/L;总胆固醇 2.27 mmol/L;
    # Assessment: 心力衰竭,冠状动脉粥样硬化性心脏病,心房颤动,心功能Ⅲ级,三尖瓣疾病重度关闭不全,脑梗死后遗症,贫血
    # Plan: 患者病情稳定，治疗方案同前，继观。 记录医生: 罗秀英"""
    r = Seg.raw_seg(s, mode='basic', adhesion=False)
    print(r)
