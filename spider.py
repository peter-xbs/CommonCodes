# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/23 3:53 下午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""

# _*_ coding:utf _*_

"""
收集部分资料
"""
import re
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


class SpiderConfig(object):
    _USER_AGENTS = [
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 LBBROWSER",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
        "Mozilla/5.0 (iPad; U; CPU OS 4_2_1 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8C148 Safari/6533.18.5",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:2.0b13pre) Gecko/20110307 Firefox/4.0b13pre",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10"
    ]
    _sleep_time_list = list(np.array(range(0, 50, 5)) / 10000)

    @classmethod
    def get_header(cls):
        return {
            # 'Accept': 'text/html,application/xhtml+xml,application/xml',
            # 'Accept-Encoding': 'gzip, deflate, compress, br',
            # 'Accept-Language': 'zh-CN,zh;q=0.9',
            # 'Cache-Control': 'max-age=0',
            # 'Connection': 'keep-alive',
            'user-agent': random.choice(cls._USER_AGENTS)
        }
    @classmethod
    def get_sleep_time(cls):
        return random.choice(cls._sleep_time_list)


class SpiderBase1(object):
    """
    动态爬虫
    """

    def __init__(self):
        chrome = '/Users/peters/Tools/chromedriver'
        self.driver = webdriver.Chrome(executable_path=chrome)
        self.driver.maximize_window()

    def terminate_driver(self):
        self.driver.quit()

    def process(self, *args):
        raise NotImplementedError


class JingDongPaiMai(SpiderBase1):
    def __init__(self):
        super().__init__()

    def process(self):
        url = 'https://auction.jd.com/sifa_list.html?provinceId=7'
        self.driver.get(url)
        res = self.driver.find_elements_by_xpath('/html/body/div[6]/div/div[4]/ul/li/a')
        for item in res:
            print(item.get_attribute('href'))
        self.terminate_driver()


class YunNanYiBaoSpider(SpiderBase1):
    def __init__(self):
        super().__init__()
        self.xpath = '//*[@id="mainBody"]/div/table/tbody/tr'
        self.base_url = "http://ylbz.yn.gov.cn/index.php?s=form&c=ylfwxm&m=page&page={}"
        self.pages = 276

    def process(self, output1, output2):
        line_list = []
        for i in range(self.pages):
            p = i + 1
            url = self.base_url.format(p)
            results = self.get_results(url)
            line_list.extend(results)
        self.terminate_driver()

        with open(output1, 'w') as fo:
            json.dump(line_list, fo, ensure_ascii=False, indent=4)
        df = pd.DataFrame(line_list, columns=['项目编号', '项目名称', '项目内涵', '除外内容', '计价单位', '说明'])
        df.to_excel(output2)

    def get_results(self, url):
        results = []
        try:
            self.driver.get(url)
            time.sleep(0.1)
            res = self.driver.find_elements_by_xpath(self.xpath)
            print('okay')
            for elem in res:
                r = (elem.find_elements_by_tag_name('td'))
                r = [e.text for e in r]
                results.append(r)

        except Exception as e:
            print(e)
            return []
        return results

class NHSA_SuppliesSpider(SpiderBase1):
    """
    医保医用耗材爬虫
    """

    def __init__(self, url, version):
        super(NHSA_SuppliesSpider, self).__init__()
        self.row_xpath = '/html/body/section/div[3]/div[3]/div[3]/div/table/tbody/tr'
        self.next_page_xpath = '//*[@id="next_gridpage"]'
        self.url = url
        self.out = '医保医用耗材_v{}.txt'.format(version)

    def crawl(self, total_pages):
        self.init_crawl()
        time.sleep(0.5)
        idx = 0
        with open(self.out, 'w') as fo:
            while idx < total_pages:
                time.sleep(0.5)
                lines = self.get_results()
                for line in lines:
                    fo.write(line + '\n')
                time.sleep(0.1)
                self.nxt_page()
                idx += 1
        self.terminate_driver()

    def init_crawl(self):
        self.driver.get(self.url)

    def get_results(self):
        lines = []
        flag = True
        rows = self.driver.find_elements_by_xpath(self.row_xpath)
        for row in rows:
            try:
                line = str(row.text)
            except:
                flag = False
                break
            lines.append(line)
        if not flag:
            lines = []
            time.sleep(0.5)
            rows = self.driver.find_elements_by_xpath(self.row_xpath)
            for row in rows:
                try:
                    line = str(row.text)
                except:
                    print('ERROR LINE...')
                    continue
                lines.append(line)
        return lines

    def nxt_page(self):
        button = self.driver.find_element_by_xpath(self.next_page_xpath)
        # print(button.text)
        # button = WebDriverWait(self.driver, 5).until(EC.is_enabled((By.XPATH, self.next_page_xpath)))
        if button.is_enabled():
            button.click()
            return True
        return False

class NHSA_SuppliesDetailSpider():
    def __init__(self, version=None, records_num=None):
        self.version = version if version else ''
        self.records_num = records_num if records_num else 0

        self.data_list_url = 'http://code.nhsa.gov.cn:8000/hc/stdPublishData/getStdPublicDataList.html?'
        self.data_detail_url = 'http://code.nhsa.gov.cn:8000/hc/stdPublishData/getStdPublicDataListDetail.html?'
class NHSA_SuppliesSpider(SpiderBase1):
    """
    医保医用耗材爬虫
    """

    def __init__(self, url, version):
        super(NHSA_SuppliesSpider, self).__init__()
        self.row_xpath = '/html/body/section/div[3]/div[3]/div[3]/div/table/tbody/tr'
        self.next_page_xpath = '//*[@id="next_gridpage"]'
        self.url = url
        self.out = '医保医用耗材_v{}.txt'.format(version)

    def crawl(self, total_pages):
        self.init_crawl()
        time.sleep(0.5)
        idx = 0
        with open(self.out, 'w') as fo:
            while idx < total_pages:
                time.sleep(0.5)
                lines = self.get_results()
                for line in lines:
                    fo.write(line + '\n')
                time.sleep(0.1)
                self.nxt_page()
                idx += 1
        self.terminate_driver()

    def init_crawl(self):
        self.driver.get(self.url)

    def get_results(self):
        lines = []
        flag = True
        rows = self.driver.find_elements_by_xpath(self.row_xpath)
        for row in rows:
            try:
                line = str(row.text)
            except:
                flag = False
                break
            lines.append(line)
        if not flag:
            lines = []
            time.sleep(0.5)
            rows = self.driver.find_elements_by_xpath(self.row_xpath)
            for row in rows:
                try:
                    line = str(row.text)
                except:
                    print('ERROR LINE...')
                    continue
                lines.append(line)
        return lines

    def nxt_page(self):
        button = self.driver.find_element_by_xpath(self.next_page_xpath)
        # print(button.text)
        # button = WebDriverWait(self.driver, 5).until(EC.is_enabled((By.XPATH, self.next_page_xpath)))
        if button.is_enabled():
            button.click()
            return True
        return False
        self.out = '医保医用耗材详情_v{}.xlsx'.format(version)

        self.list_postfix = 'releaseVersion={version}&specificationCode=&commonname=&companyName=&catalogname1=&catalogname2=&catalogname3=&_search=false&nd={nd}&rows={rows}&page={page}&sidx=&sord=asc'
        self.detail_postfix = 'specificationCode={code}&releaseVersion={version}&_search=false&rows=150&page=1&sidx=&sord=asc'

    def crawl(self, page_rows=50):
        columns = []
        lines = []
        nd = self.timestamp()
        pool = tp(10)
        page_nums = self.records_num // page_rows + 1
        print(page_nums)
        for i in range(1, page_nums + 1):
            url = self.data_list_url + self.list_postfix.format(version=self.version, nd=nd, rows=page_rows, page=i)
            r = requests.get(url, headers=SpiderConfig.get_header()).json()
            rows = r['rows']
            codes = [row['specificationCode'] for row in rows]
            for res, cols in pool.imap_unordered(self.parallel, codes):
                lines.extend(res)
                columns = cols
            print('progress {} page...'.format(i))
            print(lines[-1])
            print(len(lines))
            time.sleep(0.1)

        df = pd.DataFrame(lines, columns=columns)
        df.to_excel(self.out)

    @classmethod
    def timestamp(cls):
        return int(round(time.time() * 1000))

    def parallel(self, code):
        lines = []
        columns = []
        sub_url = self.data_detail_url + self.detail_postfix.format(code=code, version=self.version)
        sub_r = requests.get(sub_url, headers=SpiderConfig.get_header()).json()
        sub_rows = sub_r["rows"]
        for sub_row in sub_rows:
            columns = list(sub_row.keys())
            lines.append(list(sub_row.values()))
        time.sleep(0.1)
        return lines, columns


class YiMaiTongJianYanZhuShouSpider(SpiderBase1):
    def __init__(self):
        super(YiMaiTongJianYanZhuShouSpider, self).__init__()
        self.base_url = 'http://inspects.medlive.cn/'
        self.division_xpath = '//*[@id="id_f_page_cont_nav"]/ul/li'
        self.parts = ['chemistry', 'physics', 'report']
        self.item_xpath = '//*[@id="id_f_page_cont_box"]/table/tbody/tr/td'

    def crawl(self):
        for idx in self.parts:
            output = '检查助手-{}'.format(idx)
            self._crawl(idx, output)
        self.terminate_driver()

    def _crawl(self, idx, output):
        with open(output, 'w') as fo:
            division_urls, division_texts = self.get_division(idx)
            for div_url, div_text in zip(division_urls, division_texts):
                dic = {}
                item_url = self.base_url + div_url
                sub_urls, item_texts = self.get_items(item_url)
                details = []
                for sub_url, sub_text in zip(sub_urls, item_texts):
                    sub_dic = {}
                    self.driver.get(self.base_url + sub_url)
                    html = self.driver.page_source
                    time.sleep(0.2)
                    sub_dic['sub_url'] = sub_url
                    sub_dic['sub_text'] = sub_text
                    sub_dic['html'] = html
                    details.append(sub_dic)
                dic['div_url'] = div_url
                dic['div_text'] = div_text
                dic['details'] = details
                line = json.dumps(dic) + '\n'
                fo.write(line)

            def get_division(self, item):
                """
                大类
                :return:
                """
                self.driver.get(self.base_url + '?action={}'.format(item))
                time.sleep(1)
                divisions = self.driver.find_elements_by_xpath(self.division_xpath)
                res = [item.get_attribute('innerHTML') for item in divisions]
                url_res = [re.findall('href="(.*?)"', item)[0] for item in res]
                url_res = [item.replace('&amp;', '&') for item in url_res]
                text_res = [item.text for item in divisions]
                assert len(url_res) == len(text_res)
                return url_res, text_res

            def get_items(self, url):
                self.driver.get(url)
                items = self.driver.find_elements_by_xpath(self.item_xpath)
                res = [item.get_attribute('innerHTML') for item in items]
                url_res = [re.findall('href="(.*?)"', item)[0] for item in res]
                url_res = [item.replace('&amp;', '&') for item in url_res]
                text_res = [item.text for item in items]
                assert len(url_res) == len(text_res)
                return url_res, text_res

            class NHSA_YiLiaoFuWuSpider():
                """
                医保局全国医疗服务项目爬虫
                """

                def __init__(self, out, type='医技诊疗'):
                    self.out = out
                    if type == '医技诊疗':
                        self.url = 'http://code.nhsa.gov.cn:8000/ylfw/stdMedicalService/getPublicStdMedicalServiceSubTreeData.html?&page=1&sord=asc&rows=2063&msId=a68bfb8b-7a09-11e9-910a-8cec4bd010f3'
                    elif type == '临床诊疗':
                        self.url = 'http://code.nhsa.gov.cn:8000/ylfw/stdMedicalService/getPublicStdMedicalServiceSubTreeData.html?&page=1&sord=asc&rows=5569&msId=a692922f-7a09-11e9-910a-8cec4bd010f3'
                    elif type == '全部':
                        self.url = 'http://code.nhsa.gov.cn:8000/ylfw/stdMedicalService/getPublicStdMedicalServiceSubTreeData.html?&page=1&sord=asc&rows=8204&msId=0'

                def crawl(self):
                    r = requests.post(self.url, headers=SpiderConfig.get_header()).json()
                    rows = r['rows']
                    columns = []
                    lines = []
                    for row in rows:
                        columns = row.keys()
                        values = row.values()
                        lines.append(values)
                    df = pd.DataFrame(lines, columns=columns)
                    df.to_excel(self.out)

                @classmethod
                def timestamp(cls):
                    return int(round(time.time() * 1000))

class NHSA_DRUG_Spider(object):
    def __init__(self, out):
        self.url = 'http://code.nhsa.gov.cn:8000/yp/getPublicGoodsDataInfo.html?companyNameSc=&registeredProductName=&approvalCode=&batchNumber={version}&_search=false&nd={nd}&rows={rows}&page={page}&sidx=t.goods_code&sord=asc'
        self.records_num = 144616
        self.version = "20210225"
        self.out = out

    def crawl(self, page_rows=50):
        nd = self.timestamp()
        # pool = tp(10)
        fo = open(self.out, 'w')
        page_nums = self.records_num // page_rows + 1
        print(page_nums)
        for i in range(1, page_nums + 1):
            url = self.url.format(version=self.version, nd=nd, rows=page_rows, page=i)
            r = requests.get(url, headers=SpiderConfig.get_header()).json()
            rows = r['rows']
            line = json.dumps(rows, ensure_ascii=False) + '\n'
            fo.write(line)
            # rows = r['rows']
            # codes = [row['specificationCode'] for row in rows]
            # for res, cols in pool.imap_unordered(self.parallel, codes):
            #     lines.extend(res)
            #     columns = cols
            # print('progress {} page...'.format(i))
            # print(lines[-1])
            # print(len(lines))
            time.sleep(0.1)

        fo.close()

    @classmethod
    def timestamp(cls):
        return int(round(time.time() * 1000))

class LoincSpider(SpiderBase1):
    def __init__(self):
        super().__init__()
        self.src = '../data/Loinc2.7.2.csv'

    def process(self, *args):
        fin_set = set()
        lines = []
        cand_set = set()
        with open('loinc_test.txt') as f:
            for line in f:
                line = line.replace(', nan', ", ''")
                line = line.replace(', None', ", ''")
                l = eval(line.strip())
                if l[-6]:
                    lines.append(l)
                else:
                    cand_set.add(l[0])
                assert len(l) == 51
                # fin_set.add(l[0])
        # fo = open('loinc_test.txt', 'a')
        base_url = 'https://loinc.org/{}/'
        df = pd.read_csv(self.src, header=0)
        cols = list(df.columns) + ["lcn", "pt_dscrp", "grp", "grp_descrp", "zh_info", "lp_res"]

        for line in df.itertuples():
            _, *args = line
            code = args[0]
            # if code in fin_set:
            #     continue
            if code not in cand_set:
                continue
            url = base_url.format(code)
            try:
                line_ = self.get_result(url)
            except:
                line_ = [None]*6
            new_line = args + line_
            # fo.write(str(new_line)+'\n')
            lines.append(new_line)
        df = pd.DataFrame(lines, columns=cols)
        df.to_excel('LOINC_zh.xlsx', index=False)
        # fo.close()
        self.terminate_driver()

    def get_result(self, url):
        self.driver.get(url)
        time.sleep(random.choice([0.1, 0.2, 0.15, 0.18]))
        try:
            lcn = self.driver.find_elements_by_xpath('//*[@id="lcn"]')
        except:
            lcn = None
        if lcn:
            lcn_res = str(lcn[0].text)
        else:
            lcn_res = None
        try:
            pt_dscrp = self.driver.find_elements_by_xpath('//*[@id="part-descriptions"]/p')
        except:
            pt_dscrp = None
        if pt_dscrp:
            pt_dscrp_res = str(pt_dscrp[0].text)
        else:
            pt_dscrp_res = None
        try:
            lp_url = self.driver.find_elements_by_xpath('//*[@id="part-descriptions"]/p/a')
        except:
            lp_url = None
        if lp_url:
            elem = lp_url[0]
            lp_url_res = elem.get_attribute('href')
        else:
            lp_url_res = None
        try:
            grp = self.driver.find_elements_by_xpath('//*[@id="code2"]/a')
        except:
            grp = None
        if grp:
            grp_res = str(grp[0].text)
        else:
            grp_res = None
        grp_descrp = self.driver.find_elements_by_xpath('//*[@id="member-of-groups"]/table/tbody/tr/td[2]')
        if grp_descrp:
            grp_descrp_res = str(grp_descrp[0].text)
        else:
            grp_descrp_res = None
        zh_info = self.driver.find_elements_by_xpath('//*[@id="language-variants"]/dl/dd[1]')
        if zh_info:
            zh_info_res = str(zh_info[0].text)
        else:
            zh_info_res = None
        if lp_url_res:
            lp_res = self.get_lp_res(lp_url_res)
        else:
            lp_res = None
        line = [lcn_res, pt_dscrp_res, grp_res, grp_descrp_res, zh_info_res, lp_res]
        return line

    def get_lp_res(self, lp_url):
        self.driver.get(lp_url)
        syn_info = self.driver.find_elements_by_xpath('//*[@id="language-variants"]/dl/dd[1]')
        if syn_info:
            elem = syn_info[0]
            return str(elem.text)
        return ''

if __name__ == '__main__':
    spider = LoincSpider()
    spider.process()
    # spider = JingDongPaiMai()
    # spider.process()
    # 医保目录爬取
    # spider = NHSA_DRUG_Spider('医保目录_20210225.json')
    # spider.crawl()
    # # 全国医疗服务项目
    # spider = NHSA_YiLiaoFuWuSpider('医疗服务项目-全国版.xlsx', type='全部')
    # spider.crawl()
    # # 医保局耗材 含公司信息
    # url = 'http://code.nhsa.gov.cn:8000/hc/stdPublishData/toQueryStdPublicDataList.html?releaseVersion=20210116?batchNumber=20210116'
    # version='20210116'
    # spider = NHSA_SuppliesDetailSpider(version=version, records_num=40000)
    # spider.crawl(page_rows=200)
    # # 医脉通检查助手
    # spider = YiMaiTongJianYanZhuShouSpider()
    # spider.crawl()
    # 医保局耗材
    # url = 'http://code.nhsa.gov.cn:8000/hc/stdSpecification/toStdSpecificationList.html' ## 2021年1月
    # version = '2021-01'
    # url = 'http://code.nhsa.gov.cn:8000/hc/stdPublishData/toQueryStdPublicDataList.html?releaseVersion=20201024?batchNumber=20201024'
    # version = '2020-11'
    # spider = NHSA_SuppliesSpider(url, version)
    # spider.crawl()
    # 云南医保
    # spider = YunNanYiBaoSpider()
    # spider.process('test.json', '云南省医保服务项目.xlsx')
    # r = requests.get()