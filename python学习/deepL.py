import time

import requests
headers = {'Content-Type':"application/x-www-form-urlencoded",'user-agent':'Mozilla/5.0 (X11; Linux x86_64) '
                                                                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                          'Chrome/100.0.4692.36 Safari/537.36'}
def sougou1(text,f = "auto", to = "zh-CHS"):
    import pydeepl
    text = str(text)

    # ans = pydeepl.translate(text,"auto",json=True)
    # print(ans)
    url = f"https://fanyi.sogou.com/text?keyword={text}&transfrom={f}&transto={to}&model=general"
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
    browser = webdriver.Chrome()
    browser.get(url)
    element = WebDriverWait(browser, 5, 0.5).until(EC.visibility_of_element_located((By.ID,"trans-result")))

    print("搜狗翻译：" + element.text)

    # beautifulsoup解析
    html = browser.page_source
    soup = BeautifulSoup(html,"lxml")
    print(soup.find(id = 'trans-result').text)
    browser.close()

def sougou(text,f = "auto", to = "zh-CHS"):
    import pydeepl
    text = str(text)

    # ans = pydeepl.translate(text,"auto",json=True)
    # print(ans)
    url = f"https://fanyi.sogou.com/text?keyword={text}&transfrom={f}&transto={to}&model=general"

    import os

    PATH = os.environ['PATH']
    PATH = "/opt/google/chrome-beta/:" + PATH
    os.environ["PATH"] = PATH

    reponse = requests.get(url,headers=headers,timeout=(3,27))

    # beautifulsoup解析
    html = reponse.text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html,"lxml")
    print("搜狗翻译：" + soup.find(id = 'trans-result').text)





if __name__ == '__main__':
    sougou1("""In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 In the preceding example, the service layer consists of the PetStoreServiceImpl class and two data access objects of the types JpaAccountDao and JpaItemDao (based on the JPA Object-Relational Mapping standard). The property name element refers to the name of the JavaBean property, and the ref element refers to the name of another bean definition. This linkage between id and ref elements expresses the dependency between collaborating objects. For details of configuring an object’s dependencies, see Dependencies.
 
 """)