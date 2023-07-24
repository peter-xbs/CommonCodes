# _*_ coding:utf-8 _*_
# @Time: 2023-07-24 11:14
# @Author: cmcc
# @Email: xinbao.sun@hotmail.com
# @File: server_dynamic_spider.py
# @Project: CommonCodes

"""
百度医典爬虫，动静结合典型，服务器chrome支撑
"""
class SpiderBase1(object):
    """
    动态爬虫
    """

    def __init__(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

        ## options设置
        options = Options()
        chrome = '/mnt/data3/apps/LLM-prompt-data/tools/chromedriver'
        service = Service(chrome)

        ### 下载位置设置
        download = '/mnt/data3/apps/LLM-prompt-data/Downloads'
        prefs = {
            "download.default_directory": download,
            "plugins.always_open_pdf_externally": True
        }
        ### 基本设置
        #         options.add_argument("--headless")
        #         options.add_argument("--disable-gpu")
        #         options.add_argument("--no-sandbox")
        # # 禁用PDF插件，确保PDF文件直接下载而不是在浏览器中打开
        #         options.add_argument("--disable-extensions")
        #         options.add_argument("--v=1")
        # 于禁用浏览器加载和显示网页中的图像。这个设置的主要作用是在网络连接较慢或需要节省带宽的情况下加快页面加载速度。禁用图像加载可以减少
        # options.add_argument('blink-settings=imagesEnabled=false')
        # 在某些特定的系统环境下，如Docker容器中或某些虚拟化环境中，共享内存的使用可能会遇到限制或不兼容的问题。这时，通过添加--disable-dev-shm-usage选项来禁用共享内存的使用，可以解决这些问题
        # options.add_argument('--disable-dev-shm-usage') # --disable-dev-shm-usage

        ### 网络日志设置
        #         options.add_argument('--log-level=0') # 禁用日志输出到控制台
        #         options.add_argument('--enable-logging')
        #         options.add_argument('--log-level=0') # 设置日志详细级别
        #         options.add_experimental_option('perfLoggingPrefs', { 'enableNetwork': True,'enablePage': False})
        options.add_argument('--remote-debugging-port=9222')
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
        options.add_experimental_option("prefs", prefs)
        self.driver = webdriver.Chrome(service=service, options=options)

    def terminate_driver(self):
        self.driver.quit()

    def process(self, *args):
        raise NotImplementedError


class BDYD(SpiderBase1):
    def __init__(self):
        super().__init__()

    #         self.set_cookie_test(cookie)

    def process(self, url):
        self.driver.get(url)
        time.sleep(1.5)
        time.sleep(SpiderConfig.get_sleep_time())
        return self.driver.page_source


from urllib.parse import urlencode

payload = {
    'tabType': "1",
    'navType': "3",
    'itemType': ""
}

url = 'https://jiankang.baidu.com/widescreen/api/entitylist?tabType=1&navType=3'

rsp = requests.get(url, headers=SpiderConfig.get_header())

html = """<div class="_3ekmQ"><span class="_3BGVZ">类型</span><span class="mQUou"><span class="MupMp">皮肤性病科</span><span class="">感染科</span><span class="">妇产科</span><span class="">消化内科</span><span class="">内分泌科</span><span class="">骨科</span><span class="">精神心理科</span><span class="">心血管内科</span><span class="">神经内科</span><span class="">普通外科</span><span class="">耳鼻咽喉头颈外科</span><span class="">泌尿外科</span><span class="">风湿免疫科</span><span class="">儿科</span><span class="">眼科</span><span class="">呼吸内科</span><span class="">口腔科</span><span class="">神经外科</span><span class="">急诊科</span><span class="">肿瘤科</span><span class="">肝胆外科</span><span class="">血管外科</span><span class="">心胸外科</span><span class="">肛肠外科</span><span class="">血液内科</span><span class="">肾脏内科</span></span></div>"""

items = BS(html).get_text('\n')

items = items.split('\n')[1:]

base_url = 'https://jiankang.baidu.com/widescreen/api/entitylist?'
with open('百度医典_meta.jsonl', 'w') as fo:
    for item in items:
        payload['itemType'] = item
        suffix = urlencode(payload)
        url = base_url + suffix
        rsp = requests.get(url, headers=SpiderConfig.get_header())
        line = json.dumps(rsp.json(), ensure_ascii=False) + '\n'
        fo.write(line)
        time.sleep(SpiderConfig.get_sleep_time())

with open('百度医典_meta.jsonl') as f:
    url_set = set()
    for line in f:
        js = json.loads(line)
        for k in js['entityList']:
            for sd in js['entityList'][k]:
                url_set.add(sd['url'])

BD.terminate_driver()
BD = BDYD()

with open('百度医典.jsonl', 'w') as fo:
    from tqdm import tqdm

    for url in tqdm(url_set):
        try:
            rsp = BD.process(url)
            cur = {'url': url, 'html': rsp}
            line = json.dumps(cur, ensure_ascii=False) + '\n'
            fo.write(line)
        except:
            err = traceback.format_exc()
            print(err)
            BD.terminate_driver()
            BD = BDYD()
            time.sleep(3)