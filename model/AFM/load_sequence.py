import random
from collections import defaultdict
import numpy as np
class Data_Factory():
    def __init__(self, ratio,path):
        self.path = path
        self.trainfile = self.path +"train_user.txt"
        self.testfile = self.path + "test_user.txt"
        self.file = self.path + "dataset_out_2.txt"
        self.split_user(ratio, path)
        self.user_field_M, self.item_field_M, self.wordn, self.max_l = self.get_length()
        print("user_field_M", self.user_field_M)
        print("item_field_M", self.item_field_M)
        self.user_bind_M,self.item_bind_M = self.bind_user_item() #assaign a userID and item ID for a specific user-context
        print("item_bind_M", len(self.binded_items.values()))
        print("user_bind_M", len(self.binded_users.values()))
        self.user_positive_list = self.get_positive_list(self.trainfile) #userID positive itemID
        self.user_positive_list_test = self.get_positive_list(self.testfile)  # userID positive itemID
        self.clc_index, self.history_list, self.history_id, self.clc_number = self.construct_history(self.trainfile)
        self.features_M = self.map_features()
        self.Train_data,  self.Test_data, self.All_data= self.construct_data()
        self.school_list = self.construct_clc2school()

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        print("features_M:", len(self.features))
        return len(self.features)

    def get_length(self):  # 得到user和item的最大长度
        length_user = {}
        length_item = {}
        length_word = 0
        max_l = 0
        f = open(self.file, encoding='utf-8')
        line = f.readline()
        while line:
            user_features = line.strip().split(';')
            for i in range(len(user_features)):
                if i==0 or  i==5 or i==6:
                    feature = int(user_features[i])
                    if not feature in length_user:
                        length_user[feature]=1
                if i==1 or i==2:
                    feature = int(user_features[i])
                    if not feature in length_item:
                        length_item[feature]=1
                if i==3:
                    word_nums = user_features[i].strip().split(' ')
                    max_l = len(word_nums) if max_l < len(word_nums) else max_l
                    for word in word_nums:
                        length_word = int(word) if length_word < int(word) else length_word
            line = f.readline()
        f.close()
        print(max_l)
        print(length_word)
        return len(length_user), len(length_item), length_word+1, max_l

    def read_features(self, file): # read a feature file
        f = open(file,encoding='utf-8')
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(';')
            for j in range(len(items)):
                if j!=0 and j!=4:
                    if items[j] not in self.features:
                        self.features[items[j]] = i
                        i = i + 1
            line = f.readline()
        f.close()
    def bind_user_item(self):  # bind item and feature
        self.binded_users = {}
        self.binded_items = {} #dic{feature: id}
        self.item_map={}   #dic{id: feature}
        self.clc2school={}
        self.bind_u_i(self.trainfile)
        self.bind_u_i(self.testfile)
        return len(self.binded_users),len(self.binded_items)

    def bind_u_i(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        i = len(self.binded_items)
        j = len(self.binded_users)
        while line:
            features = line.strip().split(';')
            item_features = features[1]+';'+features[2]+';'+features[3]
            user_features = features[0]+';'+features[5]+';'+features[6]
            if file.__contains__('train'):
                if features[5]+';'+features[2] in self.clc2school:
                    self.clc2school[features[5]+';'+features[2]] = self.clc2school[features[5]+';'+features[2]]+1
                else:
                    self.clc2school[features[5] + ';' + features[2]] =1
            if item_features not in self.binded_items:
                    self.binded_items[item_features] = i
                    self.item_map[i]=item_features
                    i = i + 1
            if user_features not in self.binded_users:
                    self.binded_users[user_features] = j
                    j = j + 1
            line = f.readline()
        f.close()

        # bind file out
        bind_user_file = open('new/sjtu_data/bind_user.txt', 'wt')
        bind_book_file = open('new/sjtu_data/bind_book.txt', 'wt')

        for (k, v) in self.binded_items.items():
            bind_book_file.write(f'{k};{v}\n')
        for (k, v) in self.binded_users.items():
            bind_user_file.write(f'{k};{v}\n')

        bind_book_file.close()
        bind_user_file.close()

    def construct_clc2school(self):
        clc_set = sorted(self.clc2school.items(),key=lambda x:x[1],reverse=True)
        school_set = {}
        school_list = [0]*self.user_field_M
        for key in clc_set:
            keys = key[0].split(';')
            if keys[0] not in school_set:
                school_set[keys[0]]=keys[1]
        for key in school_set:
            school_list[int(key)]=int(school_set[key])-self.item_bind_M
        return school_list

    def get_positive_list(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        user_positive_list = {}
        while line:
            features = line.strip().split(';')
            user_id = self.binded_users[features[0]+';'+features[5]+';'+features[6]]
            item_id = self.binded_items[features[1]+';'+features[2]+';'+features[3]]
            if user_id in user_positive_list:
                user_positive_list[user_id].append(item_id)
            else:
                user_positive_list[user_id] = [item_id]
            line = f.readline()
        f.close()
        return user_positive_list

    def construct_data(self):
        X_user, X_item = self.load_dataset(self.trainfile)
        Train_data = self.construct_dataset(X_user, X_item)
        print("# of training:" , len(X_user))
        X_user, X_item = self.load_dataset(self.testfile)
        Test_data = self.construct_dataset(X_user, X_item)
        print("# of test:", len(X_user))
        item_list = self.binded_items.keys()
        All_items = []
        All_data = {}
        for item_key in item_list:
            item_feature = []
            it = item_key.split(';')
            item_feature.append(int(it[0]))
            item_feature.append(int(it[1]))
            item_feature.append(it[2])
            All_items.append(item_feature)
        indexs=range(len(All_items))
        All_data['X_item'] = [All_items[i] for i in indexs]
        return Train_data, Test_data,All_data
    def construct_dataset(self,X_user,X_item):
        Data_Dic = {}
        indexs =range(len(X_user))
        Data_Dic['X_user'] = [X_user[i] for i in indexs]
        Data_Dic['X_item'] = [X_item[i] for i in indexs]
        return Data_Dic

    def split_user(self, ratio, path):
        print("Randomly splitting rating sequence data into training set (%.1f) and test set (%.1f)..." % (ratio, 1 - ratio))
        trainpath = self.trainfile
        testpath = self.testfile
        rating_map = defaultdict(set)
        datafile=open(self.file, encoding='utf-8')
        trainfile = open(trainpath,'w',encoding='utf-8')
        testfile = open(testpath,'w',encoding='utf-8')

        featureLines = datafile.readlines()
        for line in featureLines:
            features= line.strip().split(';')
            key = features[0]+';'+features[5]+';'+features[6]
            value = features[1]+';'+features[2]+';'+features[3]+';'+features[4]
            rating_map[key].add(value)
        datafile.close()
        for key in rating_map.keys():
            users = key.split(';')
            values = list(rating_map[key])
            temp_data=[]
            for vs in values:
                temp_features = []
                features=vs.split(';')
                temp_features.append(features[0])
                temp_features.append(features[1])
                temp_features.append(features[2])
                temp_features.append(int(features[3]))
                temp_data.append(temp_features)
            temp_data=np.array(temp_data)
            temp_data=temp_data[np.lexsort(temp_data.T)]
            train = temp_data[:]
            test = temp_data[-1:]

            for i in range(len(train)):
                trainfile.write(
                    users[0] + ';' + ';'.join([str(j) for j in train[i]]) + ';' + users[1] + ';' + users[2])
                trainfile.write('\n')
            for i in range(len(test)):
                testfile.write(
                    users[0] + ';' + ';'.join([str(j) for j in test[i]]) + ';' + users[1] + ';' + users[2])
                testfile.write('\n')
        trainfile.close()
        testfile.close()
        print("Finish constructing training set and test set")

    #加载数据集
    def load_dataset(self,path):
        X_user, X_item=[],[]
        objfile = open(path,encoding='utf-8')
        objlines = objfile.readlines()
        for line in objlines:
            feature_line = line.strip().split(';')
            user_feature,item_feature=[],[]
            user_feature.append(int(feature_line[0]))
            #user_feature.append(int(feature_line[4]))
            user_feature.append(int(feature_line[5]))
            user_feature.append(int(feature_line[6]))
            item_feature.append(int(feature_line[1]))
            item_feature.append(int(feature_line[2]))
            item_feature.append(feature_line[3])
            X_user.append(user_feature)
            X_item.append(item_feature)
        objfile.close()
        return X_user,X_item

    def construct_history(self,path):
        objfile = open(path,encoding='utf-8')
        objlines = objfile.readlines()
        history_list={}
        history_id={}
        clc_num={}
        clc_index = [0]*(self.item_field_M - self.item_bind_M)
        for i in range(self.item_field_M - self.item_bind_M):
            clc_index[i] = i
        maxL=0
        for i in range(len(objlines)):
            feature_line = objlines[i].strip().split(';')
            key = feature_line[0]+';'+feature_line[5]+';'+feature_line[6]
            if self.binded_users[key] not in self.user_positive_list:
                continue
            if key in history_list:
                history_list[key] = history_list[key]+' '+feature_line[2]
                history_id[key] = history_id[key]+' '+feature_line[1]
            else:
                history_list[key] = feature_line[2]
                history_id[key] = feature_line[1]

        for key in history_list:
            num = np.zeros(self.item_field_M - self.item_bind_M, np.float32)
            value_str = ''
            values = history_list[key].strip().split()
            total = 0
            for value in values:
                num[int(value)-self.item_bind_M] += 1
                total += 1
                maxL = max(num[int(value)-self.item_bind_M], maxL)
            for i in range(len(num)):
                value_str += str(i)+':'+str(num[i])+' '
            clc_num[key]=value_str
        print(len(clc_num.keys()))
        self.maxL=maxL+1
        return clc_index, history_list, history_id, clc_num

if __name__=='__main__':
    df = Data_Factory(0.8, '../bslen15/')
    print(df.maxL)
