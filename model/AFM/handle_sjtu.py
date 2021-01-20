import numpy as np
import re
from tqdm import tqdm

xls='AFM/SJTU_data.txt'
outpath = 'AFM/bslen15/dataset_books.txt'
bookmap_path = 'AFM/bslen15/book_map.txt'

if __name__=='__main__':
    workbook = open(xls, 'rt')
    outfile = open(outpath, 'w', encoding='utf-8')
    bookmap_file = open(bookmap_path, 'w', encoding='utf-8')

    user_set = set()
    book_set = {}
    user_dict = {}
    grade_dict = {}
    grade_set = set()
    school_set = set()

    for line in tqdm(workbook.readlines()):
        try:
            call_no, title, author, isbn, publisher, book_year, user_name, user_id, user_school, _, _, status, date, _ = line.strip().split('@')

            # If not student with id
            if call_no == '' or isbn=='' or (len(user_id) != 12 and len(user_id)!=5) or status!='50' or date=='' or title=='' or user_school=='':
                continue

            if len(user_id)!=12 or user_id[:3]!='518':
                continue

            if not re.fullmatch('[0-9-]*', isbn):
                continue
            if len(call_no.split(' ')) >= 2:
                call_no = call_no.split(' ')[0]

            if not re.fullmatch('[0-9A-Za-z-./]*', call_no):
                continue

            if not re.fullmatch('[0-9]*', user_id):
                continue

            title = title.replace(';', '')

            grade = '0'
            if len(user_id) == 12:
                grade = user_id[1:3]

            user_set.add(user_id)

            book_str = f'{isbn};{call_no};{title}'
            user_str = f'{user_id};{user_school};{grade}'

            if user_id not in user_dict:
                user_dict[user_id] = user_str
            else:
                assert user_dict[user_id] == user_str

            if book_str not in book_set:
                book_set[book_str] = len(book_set)

            book_id = book_set[book_str]
            bookmap_file.write(f'{book_id};{book_str}\n')
            grade_set.add(grade)
            school_set.add(user_school)

            if grade not in grade_dict:
                print(grade)
                grade_dict[grade] = len(grade_dict)+1

            str1 = f'{user_id};{book_id};{call_no};{title};{date};{user_school};{grade_dict[grade]}\n'
            assert len(str1.split(';'))==7 and all(i!='' for i in str1.split(';'))

            outfile.write(str1)
        except AssertionError as err:
            print(err)
            raise AssertionError()
        except Exception as err:
            pass

    outfile.close()
    bookmap_file.close()

    print(f'book:{len(book_set)} user:{len(user_set)} grade:{len(grade_set)} school:{len(school_set)}')
# 90093 books, 20395 users, 23 grade, 81 school
