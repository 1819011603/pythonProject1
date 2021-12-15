import pymysql
import xlwt
# xlwt主要用来创建并写入数据到Excel。已经存在的表不可以写入。
# xlrd主要用来读取Excel的数据。
# 使用xlutils的copy函数来变相更改Excel的数据。  基于 xlrd/xlwt，老牌 python 包，算是该领域的先驱，功能特点中规中矩，比较大的缺点是仅支持 xls 文件。
# pandas 做数据分析 Excel 作为 pandas 输入/输出数据的容器。
# 需要进行科学计算，处理大量数据，建议 pandas+xlsxwriter 或者 pandas+openpyxl；
import os


def export_to_excel(worksheet, cursor, table):
    """
    将MySQL一个数据表导出到excel文件的一个表的函数
    :param    worksheet:  准备写入的excel表
    :param    cursor:     源数据的数据库游标
    :param    table       源数据的数据表
    :return:  Nove.
    """
    # 首先向excel表中写入数据表的字段  desc 表名；
    column_count = cursor.execute("desc %s" % table)  # 列数
    for i in range(column_count):
        temp_tuple = cursor.fetchone()  # 单行
        # print(temp_tuple)  #列属性
        # Field        | Type         | Null | Key | Default | Extra
        # id           | int unsigned | NO   | PRI | NULL    | auto_increment
        worksheet.write(0, i, temp_tuple[0])  # （0，i，temp_tuple[0]） 第0行i列 值为列名

    # 向构建好字段的excel表写入所有的数据记录  select × from table名称
    row_count = cursor.execute("select * from %s" % table)  # 行数
    for i in range(row_count):
        temp_tuple = cursor.fetchone()  # 拿出一行
        for j in range(column_count):  # 遍历每行的每列
            worksheet.write(i + 1, j, temp_tuple[j])  # (i + 1, j, temp_tuple[j])  i+1行  j列  写入i行j列的数据  第一行是列名


def database_to_excel(database, directory_name="database", sheet="sheet"):
    """

    :param database: 数据库名称
    :param directory_name: 文件夹名称
    :param sheet: excel的工作sheet名称
    """
    # 连接数据库
    connect = pymysql.Connect(
        host='127.0.0.1',  # 主机名
        port=3306,  # 端口号
        user='root',  # 用户名
        passwd='abc123',  # 密码
        charset="utf8mb4",  # 默认编码
        db=database  # 数据库名称
    )

    # 创建一个游标
    cursor = connect.cursor()
    cursor.execute("show tables")
    tables = cursor.fetchall()  # tables is tuple  fetchall()取出操作返回的所有的行
    table_list = [tuple[0] for tuple in tables]
    directory_name += "/" + database  # 文件夹路径

    if os.path.exists(directory_name) is False and os.path.isdir(directory_name) is False:
        os.makedirs(directory_name)
    path = os.getcwd() + "/" + directory_name + "/"
    for table_name in table_list:
        print("database_name: " + database + ", table_name: " + table_name + ",\tpath: " + path + table_name + ".xls")
        # 创建一个workbook 设置编码
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet(sheet)
        # 写入worksheet
        export_to_excel(worksheet, cursor, table_name)

        workbook.save(path + table_name + ".xls")

    cursor.close()
    connect.close()


if __name__ == "__main__":
    database_name = "water"  # 数据库名称
    database_to_excel(database_name)
