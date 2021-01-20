import pandas as pd                         #导入pandas包

xls = 'D:/sjtu/loandata/10247tongji/1waijie_tongji.csv'
outpath = 'D:/sjtu/loandata/10247tongji/tongji.txt'
if __name__=='__main__':
    data = pd.read_csv(xls)
    outfile = open(outpath, encoding='utf-8')
    tongji = open('D:/sjtu/loandata/10247tongji/tongji_1.txt', 'w', encoding='utf-8')
    # for i in range(1,len(data)):
    #     if data.loc[i]['PATRON_TYPE']=='本科生':
    #         temp = data.loc[i]['PATRON_ID']  用户id
    #         temp += ';'+data.loc[i]['ITEM_ID']  书籍id
    #         temp+=';'+data.loc[i]['ITEM_CALLNO']  分类号
    #         temp+=';'+data.loc[i]['TITLE']   书籍名称
    #         temp += ';' + data.loc[i]['LOAN_DATE']  借书日期
    #         temp += ';' + str(data.loc[i]['PATRON_DEPT'])   借书人专业
    #         temp += ';' + str(int(data.loc[i]['STUDENT_GRADE']))  年级
    #         outfile.write(temp)
    #         outfile.write('\n')
    # outfile.close()
    for line in outfile.readlines():
        features=line.strip().split(';')
        if len(features)!=7:
            continue
        if int(features[4].split('-')[0])<2013 or int(features[4].split('-')[0])>2016:
            continue
        tongji.write(line)
    tongji.close()