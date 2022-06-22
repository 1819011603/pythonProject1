import hashlib
import json
import random
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import QtGui
from PyQt5.QtGui import QColor, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QLabel,QTextEdit,QHBoxLayout,QVBoxLayout
import pyperclip


import sys
from PyQt5.QtCore import pyqtSlot, QMetaObject
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QVBoxLayout, QLabel,QFontDialog


pyperclip.set_clipboard("xsel")

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("翻译器")
        # 设置 (x,y,w,h)
        self.setGeometry(300, 300, 1200, 1200)

        # 设置颜色
        pal = self.palette()

        #  设置背景色为白色
        pal.setColor(self.backgroundRole(), QColor(255,255,255))
        self.setPalette(pal)

        self.edit = QTextEdit()
        self.edit.setFontFamily("SimHei")

        self.dest = "auto"
        self.dest1 = "auto"
        self.dests = ["auto","zh-CN",'en']
        self.dests_index = 0

        self.button1 = QPushButton("强制翻译", self)
        self.button2 = QPushButton("自动粘贴翻译", self)
        self.button3 = QPushButton("目标语言:" + self.dest1,self)


        self.button1.setFixedSize(400,50)
        self.button2.setFixedSize(400,50)
        self.button3.setFixedSize(400,50)



        self.auto = 0
        self.v = ['自动粘贴翻译', '不自动粘贴翻译']
        #  点击事件


        self.button1.clicked[bool].connect(self.submit)
        self.button2.clicked[bool].connect(self.submit1)
        self.button3.clicked[bool].connect(self.submit2)


        #  剪切波
        self.clipboard = QApplication.clipboard()
        # 剪切波改变
        self.clipboard.dataChanged.connect(self.on_clipboard_textChanged)

        #  设置背景 颜色
        self.edit.setStyleSheet("background-image:url(timg.jpg)")
        self.edit.setTextBackgroundColor(QtGui.QColor(0,0,0))
        self.edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))


        self.textedit = QTextEdit()


        self.textedit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self.textedit.setStyleSheet("background-image:url(timg.jpg)")

        self.baidu = QTextEdit()
        self.youdao1 = QTextEdit()
        self.edit.setObjectName("edit")


        self.textedit.move(20,45)
        self.baidu.move(20,45)
        self.youdao1.move(20,45)



        layout = QVBoxLayout()

        layout.addWidget(self.edit)

        vLayout = QHBoxLayout()

        vLayout.addWidget(self.button1)
        vLayout.addWidget(self.button2)
        vLayout.addWidget(self.button3)
        layout.addLayout(vLayout)
        layout.addWidget(self.textedit)

        layout.addWidget(self.baidu)

        self.setLayout(layout)
        QMetaObject.connectSlotsByName(self)

    def trans(self,u):
        s = u
        s = bytes(s, encoding="utf-8")  # 转成byte 才能去除 linux换行符 \n

        s = s.replace(b"-\n", b"")  # 替换
        s = s.replace(b"\n\n", b"^...^")  # 替换
        s = s.replace(b".\n", b".^^.")  # 替换
        s = s.replace(b"\n", b"")  # 替换
        s = s.replace(b"^...^", b"\n\n")  # 替换
        s = s.replace(b".^^.", b".\n")  # 替换
        s = str(s, encoding="utf-8")  # 转回str
        return s

    def trans1(self, u):
        s = u
        s = bytes(s, encoding="utf-8")  # 转成byte 才能去除 linux换行符 \n

        s = s.replace(b"-\n", b"")  # 替换
        s = s.replace(b"\n\n", b"^...^")  # 替换

        s = s.replace(b".\n", b".^^.")  # 替换
        s = s.replace(b"\n", b"")  # 替换
        s = s.replace(b"^...^", b"<br><br>")  # 替换
        s = s.replace(b".^^.", b".<br>")  # 替换
        s = str(s, encoding="utf-8")  # 转回str

        return s



    def main(self,u = None):

        if self.auto == 1:
            return None
        if u is None:
            u = str(self.clipboard.text())
        if len(u) < 3:
            return None
        text = self.trans(u)
        self.edit.setText('<font face="verdana" color="white" size="5">' +self.trans1(u)+ '</font>')
        try:
            self.textedit.setText('<font face="verdana" color="white" size="5">' +self.trans1(self.translateGoogle(text)) + '</font>')

        except Exception as e:
            self.textedit.setPlainText(str(e))
        # self.clipboard.setText(u)

        try:
            self.baidu.setText(
                '<font face="verdana" color="white" size="5">' + self.trans1(self.translateBaidu(text)) + '</font>')
        except Exception as e:
            times = 4
            while times > 0:
                times -= 1
                try:
                    self.baidu.setText('<font face="verdana" color="white" size="5">' + self.trans1(self.translateBaidu(text)) + '</font>')
                    print(times)
                    break
                except Exception as es:

                    pass





    @pyqtSlot()
    def on_edit_textChanged(self):
        #  self.edit文本改变时啥也不干
        pass



    def on_clipboard_textChanged(self):
        m =  pyperclip.paste()
        print("paste: " + m)
        self.main(u=m)

    def submit1(self):
        self.auto = 1 - self.auto
        self.button2.setText(self.v[self.auto])
        if self.auto == 0:
            self.main(u=self.edit.toPlainText())

    def submit2(self):
        self.dests_index = (self.dests_index+1 ) % len(self.dests)
        self.dest1 = self.dests [self.dests_index]
        self.button3.setText( "目标语言:" + self.dest1 )
        self.submit()

    def submit(self):
        trans = self.edit.toPlainText()
        if len(trans) < 2:
            return None
        self.edit.setText('<font face="verdana" color="white" size="5">' +self.trans1(trans) + '</font>')
        # 将光标移到最后
        self.edit.moveCursor(QTextCursor.End)
        try:
            # 设置字体
            self.textedit.setText('<font face="verdana" color="white" size="5">' +self.trans1(self.translateGoogle(trans)) + '</font>')
            # self.textedit.setPlainText(self.translateGoogle(trans))
        except Exception as e:
            self.textedit.setPlainText(str(e))
        try:
            self.baidu.setText(
                '<font face="verdana" color="white" size="5">' + self.trans1(self.translateBaidu(trans)) + '</font>')
        except Exception as e:
            times = 3
            while times > 0:
                times -= 1
                try:
                    self.baidu.setText('<font face="verdana" color="white" size="5">' + self.trans1(
                        self.translateBaidu(trans)) + '</font>')
                    break
                except Exception as es:

                    pass

    def translateGoogle(self,u):
        from googletrans import Translator

        translator = Translator()
        # 查询语言的类型
        # print(translator.detect(str(u)).lang)

        if self.dest1 == 'auto':
            if translator.detect(str(u)).lang == "zh-CN":
                self.dest = 'en'
            else:
                self.dest = 'zh-CN'
        else:
            self.dest = self.dest1


        trans_s = str(translator.translate(str(u), dest=self.dest).text)
        return trans_s

    #  f的源语言是什么语言  t是目标语言是什么语言
    def translateBaidu(self,text, f='auto', t='zh'):
        import urllib.parse, urllib.request
        import hashlib
        import urllib
        import json
        appid = '20220110001051995'
        secretKey = 'BFJctEPzjOUzQ0i21omV'
        url_baidu = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        salt = random.randint(32768, 65536)
        sign = appid + text + str(salt) + secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()

        if self.dest == 'en':
            t = self.dest
        url = url_baidu + '?appid=' + appid + '&q=' + urllib.parse.quote(
            text) + '&from=' + f + '&to=' + t + '&salt=' + str(salt) + '&sign=' + sign
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
        data = json.loads(content)
        result = str(data['trans_result'][0]['dst'])
        return result

    def exe_pdf_copyFile(self):
        from zhantieban import main,trans
        while True:
            try:
                u = str(pyperclip.paste())
                trans(u)
            except Exception as e:
                pass
            time.sleep(0.5)
        # main() # 就是一个简单的去换行
        # pass
#
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    pool = ThreadPoolExecutor(1)
    pool.submit(win.exe_pdf_copyFile)
    sys.exit(app.exec_())


