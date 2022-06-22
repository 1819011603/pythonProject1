# 剪贴板
import random

import pyperclip
import time

# pyperclip.set_clipboard("xclip")  # 这个不会输出简略信息 但是在pycharm无法删除python控制台
# pyperclip.set_clipboard("kclipper") # kde默认是klipper 但是klipper在复制长文字的时候 只显示200个长度的str 非常不方便


pyperclip.set_clipboard("xsel")  # 这个完美
text = ""
start = time.time()

import re

p1 = re.compile(r"\s作者：.*?\s*链接：.*?\s*来源：力扣（LeetCode）")
p6 = re.compile(r"[-—]{5}")
p7 = re.compile(r"著作权.*?所有.{0,2}\s*链接.*?\s\s")

end = ""

while True:
    u = str(pyperclip.paste())

    if end == u:
        time.sleep(0.5)
        continue
    end = u
    print(end)
    if re.search(p1, u) is not None:
        t = re.search(p1, u).start()

        u = u[:t]
        pyperclip.copy(u)
    if re.search(p6, u) is not None:
        t = re.search(p6, u).start()
        u = u[:t]
        pyperclip.copy(u)
    if re.search(p7,u) is not None:
        m = re.search(p7, u)
        t = m.end()
        u = u[t:]

        pyperclip.copy(u)
    time.sleep(0.5)
