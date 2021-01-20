import xlrd
import numpy as np
xls='D:/sjtu/loandata/10246复旦大学图书馆业务数据集/1图书外借数据_复旦大学.xlsx'
outpath = 'D:/sjtu/loandata/10246复旦大学图书馆业务数据集/fudan.txt'
if __name__=='__main__':
    workbook = xlrd.open_workbook(xls)
    outfile = open(outpath, 'w', encoding='utf-8')
    for k in range(4):
        work_sheet = workbook.sheet_by_index(k)
        for i in range(1, work_sheet.nrows):
            grade = work_sheet.cell(i, 25).value
            if grade.split('（')[0] == '本科生':
                str1 = work_sheet.cell(i, 22).value   
                str1 += ";" + work_sheet.cell(i, 1).value
                str1 += ";" + work_sheet.cell(i, 14).value
                str1 += ";" + work_sheet.cell(i, 17).value
                temp = int(work_sheet.cell(i, 3).value)
                str1 += ";" + str(temp)
                str1 += ";" + str(work_sheet.cell(i, 24).value)
                str1 += ";" + work_sheet.cell(i, 23).value
                outfile.write(str1 + '\n')
    outfile.close()