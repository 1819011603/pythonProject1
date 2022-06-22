import json
import time
from concurrent.futures import ThreadPoolExecutor
import pathlib
import datetime
file = "token.txt"
import requests
def test(num):
    def requestUrl(url,authorization):
        response = requests.post(url, headers={"authorization": authorization })
        # print(response.text)

    pool = ThreadPoolExecutor(max_workers=100, thread_name_prefix='测试线程')
    authorizations = []
    with open(file, "r") as f:

        for line in f.readlines():
            line = line.strip()
            authorizations.append(line.split(" ")[1])
    print(authorizations)
    start = time.time()
    times = 1
    num = num * times
    for i in range(num):
        future = pool.submit(requestUrl, "http://localhost:8080/api/voucher-order/seckill/4", authorizations[int((i/times)) % (len(authorizations))] )
    end = time.time()
    print("平均花费时间：" + str((end -start)/ (num) * 1000) + " ms.")



def getPhoneAndToken(phone):

    response = requests.post("http://localhost:8080/api/user/code?phone=" + phone)
    html = response.json()
    success = html.get("success")
    if success == False:
        return
    code = html.get("data")
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://localhost:8080/api/user/login", data=json.dumps({"phone": phone, "code": code}),
                             headers=headers)
    html = response.json()
    success = html.get("success")
    if success == False:
        return
    token = html.get("data")
    print(html)
    print(token)





    with open(file, "a+") as f:
        u = phone + " " + token + '\n'
        f.write(u)


def generate(num):
    ff = pathlib.Path(file)
    if ff.exists():
        ff.unlink()
    for i in range(100, int(num) + 100):
        phone = "13117711" + str(i)
        getPhoneAndToken(phone)

num = 100
generate(num)
test(num)
