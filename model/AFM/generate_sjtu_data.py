from tqdm import tqdm 
import re

# File Path
file_path = 'AFM/bslen15/dataset_word.txt'
output_path = 'AFM/bslen15/output.txt'
user_map_path = 'AFM/bslen15/user_map.txt'
book_map_path = 'AFM/bslen15/book_map.txt'

# Map Dicts
user_map = {}
item_map = {}
word_map = {}
school_map = {}
grade_map = {}
code_map = {}
book_correct_map = {}


def map_data(data, m):
    if m.get(data) is None:
        m[data] = len(m)

    return m[data]

fi = open(file_path, 'rt')
fo = open(output_path, 'wt')

print('Processing...')

output_list = []

for line in tqdm(fi.readlines()):
    user, book, code, title, date, school, grade = line.split(';')
    book_str = f'{book};{code};{title}'

    user_id = map_data(user, user_map)
    book_id = map_data(book_str, item_map)

    if re.match('[A-Z0-9]*', code) is not None:
        code = re.match('[A-Z0-9]*', code)[0]

    if code == '':
        continue
    code_id = map_data(code, code_map)
    title_id = [map_data(i, word_map) for i in title.split(' ')]
    school_id = map_data(school, school_map)
    grade_id = map_data(grade, grade_map)
    date = (int(date[:4])-2019) * 12 + int(date[4:6])

    if book_correct_map.get(book_str) is None:
        book_correct_map[book_str] = [code, title]
    else:
        a = book_correct_map[book_str]
        if a[0]!=code or a[1]!=title:
            print(f'Error: {book_str}')
            raise AssertionError()

    output = [user_id, book_id, code_id, " ".join(map(str, title_id)), date, school_id, grade_id]
    output_list.append(output)

print('Writing data...')
item_cnt = len(item_map)
user_cnt = len(user_map)
school_cnt = len(school_map)

for output in tqdm(output_list):
    output_str = f'{output[0]};{output[1]};{item_cnt+output[2]};{output[3]};{user_cnt+school_cnt+4+output[4]};{user_cnt+output[5]};{user_cnt+school_cnt+output[6]-1}\n'
    fo.write(output_str)

print('Saving map data...')
fi.close()
fo.close()

f_user_map = open(user_map_path, 'wt')
f_book_map = open(book_map_path, 'wt')

for (k,v) in user_map.items():
    f_user_map.write(f'{k},{v}\n')

for (k,v) in item_map.items():
    f_book_map.write(f'{k},{v}\n')

f_user_map.close()
f_book_map.close()
print('Finished!')
