import openpyxl
import os
import pathlib
class Excel2txt:
    def writelist(self,lists,txt_path):
        with open(txt_path,"w") as f:
            for item in lists:

                f.write(str(item)[1:-1] + "\n")
        txt = pathlib.Path(txt_path)

        print("写入到文件 {0} 成功".format(txt.cwd().joinpath(txt)))



    def excel2txt(self,xlsx_path = "./test_all_reflect1.xlsx"):
        intensity = openpyxl.load_workbook(xlsx_path, True).active
        rows = []
        for j,i in enumerate(intensity.rows):
            if j == 0:
                continue
            row = []
            for u in range(1,len(i)):
                row.append(i[u].value)
            rows.append(row)
        print(len(rows))
        print(len(rows[0]))
        xlsx = pathlib.Path(xlsx_path)
        print(xlsx.stem)

        self.writelist(rows,xlsx.stem + ".txt")

e = Excel2txt()
e.excel2txt()