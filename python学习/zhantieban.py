# 剪贴板
import random

import pyperclip
import time

# pyperclip.set_clipboard("xclip")  # 这个不会输出简略信息 但是在pycharm无法删除python控制台
# pyperclip.set_clipboard("kclipper") # kde默认是klipper 但是klipper在复制长文字的时候 只显示200个长度的str 非常不方便
from googletrans.models import Detected

pyperclip.set_clipboard("xsel")  # 这个完美
text = ""
start = time.time()

import re

p1 = re.compile(r"([a-zA-Z]+[0-9_]*[a-zA-Z]*\.)+[a-zA-Z]+[0-9_]*[a-zA-Z]*\((([^()\s]+\s+)*[^()\s]+)?\)")   # 代码中 含 对象.方法名(形参, 形参)

p2 = re.compile(r"import [a-zA-Z]+(\.[a-zA-Z]+)*")  # 含 import 的代码
# p3 = re.compile(r"[a-zA-Z0-9] [=|&]{2} [a-zA-Z0-9]")  # 含 == && || 的代码
# p4 = re.compile("[a-z-]\n") # 检查是否有字母/- + 换行 有才需要去换行
p5 = re.compile(r"[\u4e00-\u9fa5]{5}")  # 有中文不需要去换行
# 附加的 csdn
p6 = re.compile(r"[-—]{5}")
# leetcode
p8 = re.compile(r"(作者：\S+\s链接：)")

p7 = re.compile(r"@Override")

# 复制格式错误
p9 = re.compile("[’”，；]")

# xml
p10 = re.compile("<[a-zA-Z]*?>.*?</[a-zA-Z]*?>")

p11 = re.compile(r"((\s?([(<\"\'‘”])*[0-9a-zA-Z'’—\\◦%/-]+[,，：:.。;；]*[)>\"\'‘“]*\s?)+\s?)", re.M)

# 文件/文件夹
p12 = re.compile(r"^(file://)*(/[-.\w\u2E80-\u9FFF])+[\w.-]+")
# p11 = re.compile(@aab bbb, re.M)

# googletrans依赖的httpx 不支持 socks5代理
# 解决方案是把ALL_PROXY换为http代理:

def translate(text):
    import urllib.parse, urllib.request
    import hashlib
    import urllib
    import requests
    import sys
    import uuid
    import json


    def translateBaidu(text, f='auto', t='zh'):
        appid = '20220110001051995'
        secretKey = 'BFJctEPzjOUzQ0i21omV'
        url_baidu = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        salt = random.randint(32768, 65536)
        sign = appid + text + str(salt) + secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        url = url_baidu + '?appid=' + appid + '&q=' + urllib.parse.quote(text) + '&from=' + f + '&to=' + t + '&salt=' + str(salt) + '&sign=' + sign
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
        data = json.loads(content)
        result = str(data['trans_result'][0]['dst'])
        print("百度翻译：" + result + "\n\n")


    def youdao(text):
        url_youdao = 'https://openapi.youdao.com/api'
        APP_KEY = '1283c24b43987a6d'
        APP_SECRET = 'OuZwVaEIlJ53WZw7Tj6LPFkoURIDPDWH'

        def encrypt(signStr):
            hash_algorithm = hashlib.sha256()
            hash_algorithm.update(signStr.encode('utf-8'))
            return hash_algorithm.hexdigest()

        def truncate(q):
            if q is None:
                return None
            size = len(q)
            return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]
        salt = str(uuid.uuid1())
        curtime = str(int(time.time()))
        dicts = {'from':"AUTO",
                 'to':'zh',
                 'q':text,
                 'signType':'v3',
                 'curtime':  curtime,
                 'salt':salt,
                 'appKey':APP_KEY,
                 'sign':encrypt(APP_KEY + truncate(text) + salt + curtime + APP_SECRET)
                 }
        headers = {'Content-Type':"application/x-www-form-urlencoded",'user-agent':'Mozilla/5.0 (X11; Linux x86_64) '
                                                                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                          'Chrome/100.0.4692.36 Safari/537.36'}
        response = requests.post(url_youdao, data=dicts, headers=headers)
        import json
        json = json.loads(response.content.decode('utf-8'))
        print("有道翻译：" +json['translation'][0]+ '\n\n')

    try:
        translateBaidu( text)

    except Exception as e:
        print(f"error:{e.args}" + '\n\n')
    try:

        youdao(text)
    except Exception as e:
        print(f"error:{e.args}" + '\n\n')

    from deepL import sougou
    try:

        sougou(text)
    except Exception as e:
        print(f"error:{e.args}" + '\n\n')

while time.time() - start < 12 * 24 * 60 * 60:
    is_English = False
    u = str(pyperclip.paste())

    if text == u:
        time.sleep(0.3)
        continue

    if re.search(p6, u) is not None:

        t = re.search(p6, u).start()
        u = u[:t]
        pyperclip.copy(u)

    elif re.search(p8, u) is not None:

        t = re.search(p8, u).start()
        u = u[:t]
        pyperclip.copy(u)
    text = u

    if re.search(p12, u) is not None:
        print(f"这是一个文件/文件夹： {u}")
    elif re.search(p1, u) is None and re.search(p5, u) is None and re.search(p2, u) is None:
        p1_ans = re.search(p1, u)
        if p1_ans is None:
            print("不含有函数调用a.b(c, d)")
        else:
            print(f"含有函数调用a.b(c, d): {p1_ans.group(0)}")
        dicts = re.findall(p11, u)
        i = 0
        le = len(u)
        for d in dicts:
            i += len(d[0])
            if len(d[0]) > le * 0.80:
                is_English = True

        print(f"匹配的字符数:{i},总字符数：{le}")
        if not is_English and i > le * 0.90:
            is_English = True
        print(f"是否为英文文章： {'是' if is_English else '不是'}")

        if not is_English:
            print('\n')
            continue

        s = u
        pp = 0
        # \n 的下标list
        enter_index = [0]
        while True:
            pp = s.find('\n', pp + 1)
            if pp == -1:
                break
            enter_index.append(pp)
        print(f"换行符数组：{enter_index}")

        # 前后\n的差值
        cha_index = []

        for i in range(1, len(enter_index)):
            cha_index.append(enter_index[i] - enter_index[i - 1])

        print(f"前后换行符数组：{cha_index}")
        cha_average = 0
        # 求<120 平均值 >120我们当作是已经去好了的段落
        for i, item in enumerate(cha_index):
            if item < 120:
                cha_average += item / len(cha_index)

        # 在平均值上下15的认为应该去掉  将\n 换成 %
        for i, item in enumerate(cha_index):
            if abs(item - cha_average) < 15:
                s = s[0:enter_index[i + 1]] + '^' + s[enter_index[i + 1] + 1:]

        # 再将%换成' ' -% 换成''
        s = bytes(s, encoding="utf-8")  # 转成byte 才能去除 linux换行符 \n
        s = s.replace(b"-^", b"")  # 替换
        s = s.replace(b"^", b" ")  # 替换
        s = str(s, encoding="utf-8")  # 转回str
        pyperclip.copy(s)  # 复制回去
        u = s
        text = u
        from googletrans import Translator

        translator = Translator()
        print(u + '\n\n\n')
        trans_s = str(translator.translate(str(u), dest="zh-CN").text)
        print("谷歌翻译：" + trans_s + '\n\n')

        translate(u)
    else:
        print(f"不是英文文章！总字符数：{len(u)}"+'\n')



    time.sleep(0.3)
print("已运行12 * 24小时后关闭")


