import jieba
from tqdm import tqdm

class Manager():
    def __init__(self,length,path,prefix):
        self.length = str(length)
        self.path = path
        self.prefix = prefix
        self.data_path = self.path + self.prefix + self.length +'/dataset_books.txt'
        self.title_path = self.path + self.prefix + self.length +'/title_word.txt'
        self.word_path = self.path + self.prefix + self.length +'/dataset_word.txt'
        self.map_path = self.path + self.prefix + self.length +'/dataset_map_2.txt'
    # word cut
    def segmentation(self,str):
        seg_list = jieba.lcut(str,cut_all=False)
        return seg_list

    #filter stop words
    def getstopwords(self):
        stopwords = []
        with open("AFM/stopwords.txt", "r", encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                stopwords.append(line.strip())
        return stopwords

    def handelBookName(self, lines, outfile):
        stopwords = self.getstopwords()
        for line in lines:
            features = line.split(';')
            result_list = self.segmentation(features[3])
            sentence = list(jieba.cut(features[3]))
            sentence_segment = []
            for word in sentence:
                if word not in stopwords:
                    sentence_segment.append(word)
            temp = result_list[0]
            outfile.write(temp + '\n')

    # generate title file
    def generate_title(self):
        print('Generating Title...')
        data_file = open(self.data_path, "r", encoding='utf-8')
        lines = data_file.readlines()
        outfile = open(self.title_path, 'w', encoding='utf-8')
        count = 0
        word_map={}
        stopwords = self.getstopwords()
        max = 0
        for line in tqdm(lines):
            features = line.strip().split(';')
            if len(features) != 7:
                print('Error!!!')
                raise AssertionError()
            count += 1
            line = list(jieba.cut(features[3]))
            sentence_segment = []
            for word in line:
                if word not in stopwords:
                    if word not in word_map:
                        word_map[word]=count
                        count+=1
                    sentence_segment.append(word)
            if max < len(sentence_segment):
                max = len(sentence_segment)
            outfile.write(" ".join(sentence_segment))
            outfile.write('\n')
        print(len(word_map.keys()))
        data_file.close()
        outfile.close()

    # combine word and title
    def generate_word(self):
        print('Generating word...')
        data_file = open(self.data_path, "r", encoding='utf-8')
        title_file = open(self.title_path, "r",encoding='utf-8')
        word_file = open(self.word_path,'w',encoding='utf-8')
        datalines = data_file.readlines()
        titlelines = title_file.readlines()
        for i in tqdm(range(len(datalines))):
            split_word = titlelines[i].strip()
            features = datalines[i].strip().split(';')
            word_file.write(features[0]+';'+features[1]+';'+features[2]+';'+split_word+';'+features[4]+';'+features[5]+';'+features[6])
            word_file.write('\n')
        data_file.close()
        title_file.close()
        word_file.close()
if __name__=='__main__':
    # prefix_path = 'D:/sjtu/loandata/10247tongji/'
    prefix_path = 'AFM/'

    mg = Manager(15, prefix_path, 'bslen')
    mg.generate_title()
    mg.generate_word()
