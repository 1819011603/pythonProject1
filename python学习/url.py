import json

import requests



response  = requests.get("https://api.map.baidu.com/LBSTest/v1/?ak=ctvondSeD4TGeLibMd7ze6jpoVGmqwxD",data={

                                                            "ak":'0hYGiH3Ob5ZhV0eWzrGVXCD3bEdBCi6L',
                                                            "location": "112.950016,28.152747"
                                                        })
print(response.content.decode("utf-8"))