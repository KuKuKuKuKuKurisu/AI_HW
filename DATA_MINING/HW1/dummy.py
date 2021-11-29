import os
import time


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

def create_C1(data_set):
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1

def create_Ck(ksub1, k):
    Ck = set()
    len_ksub1 = len(ksub1)
    list_ksub1 = list(ksub1)
    for i in range(len_ksub1):
        for j in range(i+1, len_ksub1):
            l1 = list(list_ksub1[i])
            l2 = list(list_ksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_ksub1[i] | list_ksub1[j]
                Ck.add(Ck_item)
    return Ck

def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) > min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk

def generate_L(data_set, k, min_support):
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    ksub1 = C1
    L = []
    L.append(L1)
    for i in range(2, k+1):
        Ci = create_Ck(ksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        ksub1 = Ci
        L.append(Li)
    return L, support_data


def generate_big_rules(L, support_data, min_conf):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf > min_conf and big_rule not in big_rule_list:
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list

#finallist=read_csv('dataset/GroceryStore/Groceries.csv')
finallist=read_txt('dataset/UNIX_usage')
time_start=time.time()
L, support_data = generate_L(finallist, k=2, min_support=0.1)
big_rules_list = generate_big_rules(L, support_data, min_conf=0.9)
time_end=time.time()
print('totally cost',time_end-time_start)
"""
for Lk in L:
    for freq_set in Lk:
        print(freq_set, support_data[freq_set])
for item in big_rules_list:
    print (item[0], "=>", item[1], "conf: ", item[2])

"""
"""
with open("b.txt", 'w') as f:
    for i in rules:
        k = ' '.join([str(j) for j in i])
        f.write(k + "\n")
"""