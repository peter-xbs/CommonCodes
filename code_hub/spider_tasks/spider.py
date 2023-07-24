# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/23 3:53 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
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

class Kaoyan(SpiderBase1):
    def __init__(self):
        super().__init__()
    def _proc_one(self, url):
        self.driver.get(url)
        res = self.driver.find_elements(By.XPATH, '/html/body/div/div/div/div/div/div/button')
        all_htmls = []
        for item in res[:10]:
            item.click()
            time.sleep(0.5)
            id_ = item.get_attribute('id')
            id2 = id_.replace('questionbtn', 'questionall')
            iframe = self.driver.find_element(By.ID, id2)
            self.driver.switch_to.frame(iframe)
            html = self.driver.page_source
            all_htmls.append(html)
            self.driver.switch_to.default_content()
            time.sleep(0.1)
        return all_htmls

    def process(self):
        with open('../data_temp/考研.jsonl') as f, open('../data_temp/kaoyan_supp.jsonl', 'a') as fo:
            idx= 0
            for line in f.readlines():
                idx += 1

                if idx <= 1295:
                    continue
                print(idx)
                js = json.loads(line)
                url = js['type_url']
                url = url.replace('??', '?')
                js['type_url'] = url

                try:
                    htmls = self._proc_one(url)
                    status = 'ok'
                except:
                    htmls = []
                    status = 'failed'
                js['htmls'] = htmls
                js['status'] = status
                time.sleep(0.2)
                line_ = json.dumps(js, ensure_ascii=False)+'\n'
                fo.write(line_)

        self.terminate_driver()

class YMT(SpiderBase1):
    def __init__(self):
        super().__init__()

    def _proc_one(self, url):
        self.driver.get(url)
        t = 0
        all_elems = []
        while t < 70:
            self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(1)
            elems = self.driver.find_elements(By.XPATH, '//*[@class="title"]/a')
            print('ok')
            for e in elems:
                # e = elem.find_element(By.XPATH, '//*[@class="title"]/a')  # //*[@id="more"]/div[540]
                text, href = e.text, e.get_attribute('href')
                all_elems.append([text, href])
            t += 1
        return all_elems

    def process(self, url_dict):
        with open('../data_temp/医脉通guideline.jsonl', 'w') as fo:
            for topic in url_dict:
                url = url_dict[topic]
                all_elems = self._proc_one(url)
                cur = {'topic': topic, 'url': url, 'elems': all_elems}
                line = json.dumps(cur, ensure_ascii=False) + '\n'
                fo.write(line)

        self.terminate_driver()

class DXY(SpiderBase1):
    def __init__(self):
        super().__init__()

    def _proc_one(self, url):
        print(url)
        self.driver.get(url)

        all_res = []
        elems = self.driver.find_elements(By.XPATH, '//*[@id="main"]/div/div/div[1]/dl/dd/p[1]/a')
        res = [[elem.get_attribute('href'), elem.text] for elem in elems]
        all_res.extend(res)
        nxt_page_bt = self.driver.find_elements(By.XPATH, '//*[@title="下一页"]')
        print('ok', nxt_page_bt)
        while nxt_page_bt:
            href = nxt_page_bt[0].get_attribute('href')
            self.driver.get(href)
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(3)
            elems = self.driver.find_elements(By.XPATH, '//*[@id="main"]/div/div/div[1]/dl/dd/p[1]/a')
            res = [[elem.get_attribute('href'), elem.text] for elem in elems]
            time.sleep(random.choice(list(range(3))))
            nxt_page_bt = self.driver.find_elements(By.XPATH, '//*[@title="下一页"]')
            all_res.extend(res)
        return all_res

    def process(self, url_dict):
        with open('../data_temp/丁香园guideline.jsonl', 'w') as fo:
            for topic in url_dict:
                url = url_dict[topic]
                all_elems = self._proc_one(url)
                cur = {'topic': topic, 'url': url, 'elems': all_elems}
                line = json.dumps(cur, ensure_ascii=False) + '\n'
                fo.write(line)

        self.terminate_driver()

if __name__ == '__main__':
    pass