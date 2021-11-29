import time

from fpgrowth_py import fpgrowth
import os
def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    rows=rows[1:]
    for row in rows:
        row=row.split(',',1)
        row=row[1]
        row=row[2:-2]
        final_list.append(row.split(','))
    return final_list
def read_txt(file_name):
    final_list=[[]]
    folder_list=os.listdir(file_name)
    for folder in folder_list:
        folder_path=os.path.join(file_name,folder)
        file_list=os.listdir(folder_path)
        for file in file_list:
            file_path=os.path.join(folder_path,file)
            f=open(file_path,'r')
            content=f.read()
            rows=content.split('\n')
            temp_list=list()
            for row in rows:
                if row=='**SOF**':
                    pass
                elif row=='**EOF**':
                    if len(temp_list)!=0:
                        final_list.append(list(set(temp_list.copy())))
                        temp_list.clear()
                else:
                    temp_list.append(row)
    return final_list[1:]

#finallist=read_csv('dataset/GroceryStore/Groceries.csv')
finallist=read_txt('dataset/UNIX_usage')
time_start=time.time()
freqItemSet, rules = fpgrowth(finallist, minSupRatio=0.3, minConf=0.9)
time_end=time.time()
print(freqItemSet)
print('totally cost',time_end-time_start)
"""
with open("b.txt", 'w') as f:
    for i in rules:
        k = ' '.join([str(j) for j in i])
        f.write(k + "\n")
"""