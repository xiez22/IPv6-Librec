from collections import defaultdict
import re
import xlrd
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

class Factory():
    def __init__(self,length,path,prefix,level):
        self.length = str(length)
        self.path = path
        self.prefix = prefix
        self.level = level
        self.word_path = self.path + self.prefix + self.length + '/dataset_word.txt'
        self.map_path = self.path + self.prefix + self.length + '/dataset_map_'+str(level)+'.txt'
        self.emb_path = self.path + self.prefix + self.length + '/dataset_emb_' + str(level) + '.txt'
        self.out_path = self.path + self.prefix + self.length + '/dataset_out_'+str(level)+'.txt'
        self.index_path = self.path + self.prefix + self.length + '/index_level' + str(level) + '.txt'
        try:
            self.w2v = Word2Vec.load('AFM/w2v.model')
        except Exception as err:
            print('No W2V model.')
            # self.train_new_w2v()
            self.w2v = None

        try:
            f = open(self.index_path, 'rt')
        except Exception as err:
            print('No index file. Generating...')
            self.generate_index_file()

    def generate_index_file(self):
        self.index_map = {}

        with open(self.word_path, 'rt') as f:
            for line in tqdm(f.readlines()):
                code = line.split(';')[2]
                code_str = re.match('[A-Z0-9]*', code)[0]
                
                if code_str not in self.index_map:
                    self.index_map[code_str] = len(self.index_map)

    def train_new_w2v(self):
        print('Loading words...')
        word_list = []
        
        with open(self.word_path, 'rt') as f:
            for line in f.readlines():
                word_list.append(line.split(';')[3].split(' '))

        model = Word2Vec(sentences=word_list, min_count=10)
        model.save('AFM/w2v.model')

        self.w2v = model
        
    def get_keys(self, d, value):
        return [k for k, v in d.items() if v == value]

    def construct_map(self,path):
        print("constructing map about id to index ......")
        datafile = open(path, encoding='utf-8')
        datalines = datafile.readlines()
        item_id_map, user_id_map, school_id_map = {}, {}, {}
        user_base, item_base, school_base, data_base= 0, 0, 0, 0
        clc_map = {}
        clc_base = 0
        date_map = {}
        grade_map = {}
        grade_base = 0
        for line in datalines:
            features = line.strip().split(';')
            if features[5] == '':
                continue
            if features[0] not in user_id_map:
                user_id_map[features[0]] = user_base
                user_base += 1
            if features[1] not in item_id_map:
                item_id_map[features[1]] = item_base
                item_base += 1
            if features[2] not in clc_map:
                if bool(re.search('[A-Z]', features[2])):
                    print(features[2])
                clc_map[features[2]] = clc_base
                clc_base += 1
            if features[4] not in date_map:
                date_map[features[4]] = data_base
                data_base += 1
            if features[5] not in school_id_map:
                school_id_map[features[5]] = school_base
                school_base += 1
            if features[6] not in grade_map:
                grade_map[features[6]] = grade_base
                grade_base+=1
        print("user length :{0},item length :{1},date length:{2},school length :{3}".format(len(user_id_map),
                                                                                            len(item_id_map),
                                                                                            len(date_map),
                                                                                            len(school_id_map)))
        datafile.close()
        print(len(clc_map))
        return user_id_map, item_id_map, school_id_map, clc_map, date_map

    # write to out file
    def generate_out(self):
        out_map_file = open(self.out_path, 'w', encoding='utf-8')
        user_id_map, item_id_map, school_id_map, clc_map, date_map = self.construct_map(self.emb_path)
        map_file = open(self.emb_path, encoding='utf-8')
        maplines = map_file.readlines()
        print(f'用户{len(user_id_map)} 物品{len(item_id_map)} 类别{len(self.index_map)} 学院{len(school_id_map)}')
        for line in maplines:
            features = line.strip().split(';')
            out_map_file.write(str(user_id_map[features[0]]) + ';' + str(item_id_map[features[1]]) + ';' + str(
                len(item_id_map) + clc_map[features[2]]) + ';' +
                               features[3] + ';' + str(
                len(user_id_map) + len(school_id_map) + 4 + date_map[features[4]]) + ';' +
                               str(len(user_id_map) + school_id_map[features[5]]) + ';' + str(
                len(user_id_map) + len(school_id_map) + int(features[6]) - 1) + '\n')
        out_map_file.close()
        map_file.close()

        user_map_file = open('AFM/bslen15/user_map.txt', 'wt')
        for (k,v) in user_id_map.items():
            user_map_file.write(f'{k};{v}\n')
        user_map_file.close()

        book_map_file = open('AFM/bslen15/book_id_map.txt', 'wt')
        for (k,v) in item_id_map.items():
            book_map_file.write(f'{k};{v}\n')
        book_map_file.close()

    def constructMap(self):
        index_mapping = defaultdict(set)
        workbook = xlrd.open_workbook(self.index_path)
        work_sheet = workbook.sheet_by_index(0)
        for i in range(1, work_sheet.nrows):
            index_mapping[work_sheet.cell(i,0).value]=i-1
        return index_mapping

    def handle_sjtu_index(self, bookindex, index_mapping):
        bookindex_str = re.match('[A-Z0-9]*', bookindex)[0]

        return index_mapping[bookindex_str]
        

    def handlebookindex2(self,bookindex,index_mapping):
        index_first = bookindex.split('/')[0]
        if index_first.__contains__('.'):
            index_first=index_first.split('.')[0]
        if index_first[0:2] == 'E7' or index_first[0:2] == 'E3':
            return index_mapping['E3/7']
        elif index_first[0:2] == 'I7' or index_first[0:2] == 'I3':
            return index_mapping['I3/7']
        if len(index_first)<3:
            return index_mapping[index_first]
        index_first_three=index_first[0:3]
        if index_first_three=='D37' or index_first_three=='D33':
            return index_mapping['D33/37']
        if not index_mapping.keys().__contains__(index_first_three):
            if not index_mapping.keys().__contains__(index_first[0:2]):
                return bookindex
            return index_mapping[index_first[0:2]]
        return index_mapping[index_first_three]

    #first level
    def handlebookindex1(self,bookindex,index_mapping):
        index_first = bookindex.split('/')[0]
        if index_first.__contains__('.'):
            index_first=index_first.split('.')[0]
        return index_mapping[index_first[0]]

    # third level
    def handlebookindex3(self,bookindex,index_mapping):
        index_first = bookindex.split('/')[0]
        if index_first.__contains__('.'):
            index_first=index_first.split('.')[0]
        #D33/37,D73/77,E3/7,I3/7,特殊处理
        if index_first[0:2] == 'E7' or index_first[0:2] == 'E3':
            return index_mapping['E3/7']
        elif index_first[0:2] == 'I7' or index_first[0:2] == 'I3':
            return index_mapping['I3/7']
        if len(index_first)<3:
            return index_mapping[index_first]
        index_first_three=index_first[0:3]
        index_first_four = index_first[0:4]
        if index_first_three=='D37' or index_first_three=='D33':
            return index_mapping['D33/37']
        if index_mapping.keys().__contains__(index_first_four):
            return index_mapping[index_first_four]
        if not index_mapping.keys().__contains__(index_first_three):
            if not index_mapping.keys().__contains__(index_first[0:2]):
                return bookindex
            return index_mapping[index_first[0:2]]
        return index_mapping[index_first_three]

    def word2vec_transform(self, sentence):
        size = self.w2v.layer1_size

        data = sentence.split()
        length = len(data)
        vec = np.zeros(shape=(1, size), dtype=np.float32)
        for word in data:
            try:
                vec += self.w2v.wv[word]
            except:
                length -= 1
                continue
        if length==0:
            return vec
        vec = vec / length
        return vec

    def generate_map(self):
        word_file = open(self.word_path, encoding='UTF-8')
        map_file = open(self.map_path, 'w', encoding='UTF-8')
        lines = word_file.readlines()
        # index_mapping = self.constructMap()
        index_mapping = self.index_map
        index_map = {}
        # func = 'self.handlebookindex' + str(self.level)
        func = 'self.handle_sjtu_index'
        for line in lines:
            features = line.strip().split(';')
            # if int(features[6])>12:
            #     continue
            year = features[4][2:4]
            mouth = features[4][4:6]
            date_cross = (int(year) - 19) * 12 + int(mouth) - 1
            result = []
            # arr = self.word2vec_transform(features[3])[0]
            arr = features[3].split()
            for ar in arr:
                result.append(ar)
            school2index = str(eval(func)(features[2], index_mapping)) + ';' + features[5]
            if school2index in index_map:
                index_map[school2index] += 1
            else:
                index_map[school2index] = 1
            map_file.write(features[0] + ';' + features[1] + ';' + str(
                eval(func)(features[2], index_mapping)) + ';' + ' '.join(
                str(v) for v in result) + ';' + str(date_cross) + ';' + features[5] + ';' + features[6])
            map_file.write('\n')
        word_file.close()
        map_file.close()

    def generate_word_emb(self):
        word_file = open(self.word_path, encoding='UTF-8')
        emb_file = open(self.emb_path, 'w', encoding='UTF-8')
        lines = word_file.readlines()
        # index_mapping = self.constructMap()
        index_mapping = self.index_map
        index_map = {}
        word_map = {}
        index = 0
        # func = 'self.handlebookindex' + str(self.level)
        func = 'self.handle_sjtu_index'
        for line in tqdm(lines):
            features = line.strip().split(';')
            # if int(features[6])>12:
            #     continue
            year = features[4][2:4]
            mouth = features[4][4:6]
            date_cross = (int(year) - 19) * 12 + int(mouth) - 1
            result = []
            for ft in features[3].strip().split(' '):
                if ft in word_map:
                    result.append(word_map[ft])
                else:
                    word_map[ft] = index
                    result.append(index)
                    index = index+1
            school2index = str(eval(func)(features[2], index_mapping)) + ';' + features[5]
            if school2index in index_map:
                index_map[school2index] += 1
            else:
                index_map[school2index] = 1
            emb_file.write(features[0] + ';' + features[1] + ';' + str(
                eval(func)(features[2], index_mapping)) + ';' + ' '.join(
                str(v) for v in result) + ';' + str(date_cross) + ';' + features[5] + ';' + features[6])
            emb_file.write('\n')
        print(index)
        word_file.close()
        emb_file.close()

if __name__=='__main__':
    path = 'AFM/'
    prefix = 'bslen'
    level = 2
    ft = Factory(15, path, 'bslen', level)
    ft.generate_word_emb()
    ft.generate_out()
