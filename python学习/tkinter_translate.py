

import time
import tkinter.font
from concurrent.futures import ThreadPoolExecutor
from tkinter import  *


print(tkinter.font.Font)

width=1200
height=1200
bg_color= "#C9C9C9"
text_bg_color='#d3fbfb'
# text_font= ( 'hack',16)
text_font= ('黑体',12,'bold')

root = Tk()
root.wm_title("ppppp")
# root.title("简易翻译器")

root.geometry('{0}x{1}'.format(width,height))
# root.iconbitmap("")
root["background"] = bg_color
scrollbar_v = Scrollbar(root)
scrollbar_v.pack(side=RIGHT, fill=Y)
scrollbar_h = Scrollbar(root)
scrollbar_h.pack(side=RIGHT, fill=X)
text1 = Text(root,  bg=text_bg_color,
        fg='black',
        width=400,
        height=20,
        font=text_font,
        relief=SUNKEN,
        yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set
            )
text1.pack(side=BOTTOM)
# tkinter.messagebox.askokcancel("提示", "Hello,World")
text2 = Text(root,  bg=text_bg_color,
        fg='black',
        width=400,
        height=20,
        font=text_font,
        relief=SUNKEN,
        yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set
            )
text2.pack(side=TOP)
def getTextInput():
    result = text1.get("1.0", "end")  # 获取文本输入框的内容
    # print(result)  # 输出结果
    return result


# Tkinter 文本框控件中第一个字符的位置是 1.0，可以用数字 1.0 或字符串"1.0"来表示。
# "end"表示它将读取直到文本框的结尾的输入。我们也可以在这里使用 tk.END 代替字符串"end"。


# 第7步，在图形化界面上设定一个button按钮（#command绑定获取文本框内容的方法）

from zhantieban import *


def translateGoogle(u):
    from googletrans import Translator

    translator = Translator()
    print(u + '\n\n\n')
    trans_s = str(translator.translate(str(u), dest="zh-CN").text)
    return trans_s
def main():
    last=""
    while True:
        t = getTextInput()

        if last == t:
            time.sleep(0.5)
            continue
        last = t
        tex = trans(t)
        if (len(tex) == 0):
            time.sleep(0.3)
            continue
        translate(tex)
        time.sleep(0.3)
        v = translateGoogle(tex)
        root.wm_title(v)
        text2.insert("1.0",v)






pool = ThreadPoolExecutor(max_workers=1)
pool.submit(main)

# 第9步，
root.mainloop()
if __name__ == '__main__':
    main()
# from tkinter import Tk, font
# print(font.families())



