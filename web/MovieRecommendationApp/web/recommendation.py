from django.db.models.manager import BaseManager
import numpy as np 
import pandas as pd
from web.models import Myrating, Movie, Book_Data
import scipy.optimize 
import random

# Get Popular List
print('Getting popular list...')
popular_book_rec = Book_Data.objects.order_by('-borrow_cnt')
popular_book_list = []
popular_book_name_list = []
popular_book_cnt_list = []
popular_book_author_list = []

for book in popular_book_rec:
    if book.name not in popular_book_name_list:
        popular_book_name_list.append(book.name)
        popular_book_list.append(book.ISBN)
        popular_book_cnt_list.append(book.borrow_cnt)
        author_str = book.author.rstrip()
        author_str = author_str if author_str[-1]!=',' and author_str[-1]!='，' else author_str[:-1]
        popular_book_author_list.append(author_str)
    if len(popular_book_list) >= 1000:
        break

print("Finished")

class MyRecommend():
    def __init__(self, data_path, map_path):
        self.data_path = data_path
        self.map_path = map_path
        self.read_file()

    def read_file(self):
        f_data = open(self.data_path, 'rt')
        f_map = open(self.map_path, 'rt')

        self.book_map = {}
        for line in f_map.readlines():
            book_id, isbn, _, _ = line.strip().split(';')
            self.book_map[book_id] = isbn

        self.recommend_map = {}
        for line in f_data.readlines():
            user_id, recommend_list = line.strip().split(';')
            recommend_list = recommend_list.split()

            self.recommend_map[user_id] = recommend_list

    def get_recommend(self, user_id):
        if self.recommend_map.get(user_id) is None:
            return None
        else:
            return list(map(self.book_map.get, self.recommend_map[user_id]))


def getPopular(movies: BaseManager, cnt=100, choose_cnt=10):
    recommend_list = popular_book_list[:cnt]
    selected_items = []
    selected_objects = []

    while choose_cnt >= 0:
        # Random Select an Item
        select_id = random.randint(0, cnt-1)
        if select_id in selected_items:
            continue
        
        selected_items.append(select_id)
        selected_objects.append(Movie.objects.filter(ISBN__icontains=recommend_list[select_id]).distinct()[0])
        choose_cnt -= 1

    return selected_objects


def getPopularForApp(cnt=50, choose_cnt=15):
    select_isbn = []
    select_name = []
    select_author = []
    select_item_id = []

    while choose_cnt >= 0:
        select_id = random.randint(0, cnt-1)
        if select_id in select_item_id:
            continue

        select_item_id.append(select_id)
        select_isbn.append(popular_book_list[select_id])
        select_name.append(popular_book_name_list[select_id])
        select_author.append(popular_book_author_list[select_id])
        choose_cnt -= 1

    return select_isbn, select_name, select_author


def getMostPopularList(cnt=10):
    name_list = popular_book_name_list[:cnt]
    cnt_list = popular_book_cnt_list[:cnt]

    return name_list, cnt_list
