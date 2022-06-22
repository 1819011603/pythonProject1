import random
import time

import requests
headers = {'Content-Type':"application/x-www-form-urlencoded",'user-agent':'Mozilla/5.0 (X11; Linux x86_64) '
                                                                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                          'Chrome/100.0.4692.36 Safari/537.36'}
def daka(next_time):

    url = "https://ca.csu.edu.cn/authserver/login?service=https%3A%2F%2Fwxxy.csu.edu.cn%2Fa_csu%2Fapi%2Fcas%2Findex%3Fredirect%3Dhttps%253A%252F%252Fwxxy.csu.edu.cn%252Fncov%252Fwap%252Fdefault%252Findex%26from%3Dwap"
    from selenium import webdriver
    from selenium.webdriver.support.wait import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from bs4 import BeautifulSoup
    import lxml
    import os

    PATH = os.environ['PATH']
    PATH = "/opt/google/chrome-beta/:" + PATH
    os.environ["PATH"] = PATH
    os.system("unset http_proxy")
    os.system("unset http_proxys")
    browser = webdriver.Chrome("/home/lenong0427/PycharmProjects/pythonProject1/python学习/chromedriver")
    text_notice_text = "获取失败"
    try:
        browser.get(url)
        username = browser.find_element_by_id("username")
        username.clear()
        username.send_keys("204711074")
        password = browser.find_element_by_xpath('//*[@id="password"]')
        password.clear()
        password.send_keys("lovely095268")
        # time.sleep(0.5)
        login_submit= browser.find_element_by_id("login_submit")
        login_submit.click() # 登陆
        time.sleep(0.5)
        location = browser.find_element_by_xpath("//div[@name='area']")
        location.click()
        time.sleep(200)
        locations = ['湖南省 长沙市 岳麓区',"湖南省 长沙市 天心区", "湖南省 长沙市 宁乡市"]
        l = locations[int(random.randint(0,99)/20) % 3]
        js = 'document.getElementsByName("area")[0].getElementsByTagName("input")[0].removeAttribute(' \
             f'"readonly");document.getElementsByName("area")[0].getElementsByTagName("input")[0].value="{l}"; '
        browser.execute_script(js)
        time.sleep(0.5)
        submit = browser.find_element_by_xpath("//div[@class='footers']")
        submit.click()

        text_notice = browser.find_element_by_xpath('//*[@id="wapat"]//div[@class="wapat-title"]')
        text_notice_text = text_notice.text

    except Exception as e:
        print(e)


    finally:
        # browser.close()
        import datetime
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        w = text_notice_text + f"\n今天打卡时间为:{now} \n明天打卡时间为:{next_time} \nIMAP授权密码: OAWEELFKTXKRBCOW"
        print(w)
        setEmail(w )


def setEmail(content):
    import smtplib
    from email.mime.text import MIMEText
    # 设置服务器所需信息
    # 163邮箱服务器地址
    mail_host = 'smtp.163.com'
    # 163用户名
    mail_user = 'lenong0427'
    # 密码(部分邮箱为授权码)
    mail_pass = 'OAWEELFKTXKRBCOW'
    # 邮件发送方邮箱地址
    sender = 'lenong0427@163.com'
    # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
    receivers = ['1819011603@qq.com']

    # 设置email信息
    # 邮件内容设置
    message = MIMEText(content, 'plain', 'utf-8')
    # 邮件主题
    message['Subject'] = content.split("\n")[0]
    # 发送方信息
    message['From'] = sender
    # 接受方信息
    message['To'] = receivers[0]

    # 登录并发送邮件
    try:
        smtpObj = smtplib.SMTP()
        # 连接到服务器
        smtpObj.connect(mail_host, 25)
        # 登录到服务器
        smtpObj.login(mail_user, mail_pass)
        # 发送
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        # 退出
        smtpObj.quit()
        print('success')
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误



def setCircleTime(method=daka):

    import datetime
    while True:
        now = datetime.datetime.now()
        next11 = now + datetime.timedelta(days=1)
        hour = random.randint(6,11)
        minutes = random.randint(0,59)
        seconds = random.randint(0,59)
        next11 = next11.replace(hour=hour, minute=minutes,second=seconds)
        seconds = (next11- now).seconds
        method(next11.strftime('%Y-%m-%d %H:%M:%S'))

        time.sleep(seconds)










if __name__ == '__main__':
   # daka()
    setCircleTime()