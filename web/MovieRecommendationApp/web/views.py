from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render, get_object_or_404, redirect, HttpResponse
from django.db.models import Q
from django.http import Http404
from .models import Movie, Myrating, User_book, Book_Data
from django.contrib import messages
from .forms import UserForm
from django.db.models import Case, When
from .recommendation import MyRecommend, getPopular, getMostPopularList, getPopularForApp
import numpy as np
import pandas as pd
import pickle
import json
import time
import random

# Init Recommend Model
recommend_engine = MyRecommend(data_path='recommend_map.txt', map_path='book_map.txt')

# Authenticated Token Dict for Users
class Token():
    def __init__(self, user_id, expire_time=3600):
        self.user_id = user_id
        self.login_time = time.time()
        self.expire_time = expire_time
        self.token_value = str(random.randint(10**32, 10**33))
    
    def get_token(self):
        return self.token_value

    def check_valid(self):
        return time.time() < self.login_time + self.expire_time

user_token_dict = {}

def get_user_from_token(token_str):
    token: Token = user_token_dict.get(token_str)
    if token is None or not token.check_valid():
        return None
    else:
        return token.user_id

def create_token(user_id):
    token: Token = Token(user_id=user_id)
    user_token_dict[token.get_token()] = token
    return token.get_token()

# for recommendation
def recommend(request):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404
    df = pd.DataFrame(list(Myrating.objects.all().values()))
    nu = df.user_id.unique().shape[0]
    current_user_id = request.user.id
    current_studet_id = request.user.username

    ub_list = User_book.objects.filter(
        Q(student_id__icontains=current_studet_id))
    book_list = None

    if len(ub_list) > 0:
        book_list = []
        for isbn in ub_list:
            if isbn.book == '':
                continue
            book = Movie.objects.filter(Q(ISBN__icontains=isbn.book)).distinct()[0]
            book_list.append(book)
        book_list = list(set(book_list))

    # Recommendation
    rec_isbn_list = recommend_engine.get_recommend(current_studet_id)
    rec_list = []

    if rec_isbn_list:
        for isbn in rec_isbn_list:
            book = Movie.objects.filter(Q(ISBN__icontains=isbn)).distinct()[0]
            rec_list.append(book)

    return render(request, 'web/recommend.html', {'rec_list': rec_list, 'book_list': book_list})


# (App) app_recommend
def app_recommend(request):
    token = request.GET.get('test')

    user_id = get_user_from_token(token)
    if user_id is None:
        return_json = {
            "result": -1
        }
        return HttpResponse(json.dumps(return_json))

    # Recommendation
    rec_isbn_list = recommend_engine.get_recommend(user_id)
    if rec_isbn_list is None:
        return_json = {
            "result": 0
        }
        return HttpResponse(json.dumps(return_json))

    name_list = []
    author_list = []

    if rec_isbn_list:
        for isbn in rec_isbn_list:
            book = Book_Data.objects.filter(Q(ISBN__icontains=isbn)).distinct()[0]
            name_list.append(book.name)
            author = book.author
            if author[-1]==',' or author[-1]=='，':
                author = author[:-1]
            author_list.append(author)

    return_json = {
        "result": 1,
        "isbn": rec_isbn_list,
        "name": name_list,
        "author": author_list
    }

    return HttpResponse(json.dumps(return_json))


# List view
def index(request):
    movies = Movie.objects.all()
    query = request.GET.get('q')
    name_list, cnt_list = getMostPopularList(cnt=10)

    if query:
        movies = Movie.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'web/list.html', {'movies': movies, 'name_list': json.dumps(name_list), 'cnt_list': json.dumps(cnt_list)})
    else:
        # Get popular books.
        movies = getPopular(movies=movies, cnt=30, choose_cnt=10)

    return render(request, 'web/list.html', {'movies': movies, 'name_list': json.dumps(name_list), 'cnt_list': json.dumps(cnt_list)})


# (App) app_hot_recommend
def app_hot_recommend(request):
    print(request)
    isbn_list, name_list, author_list = getPopularForApp()
    return_json = {
        "isbn": isbn_list,
        "name": name_list,
        "author": author_list
    }

    return HttpResponse(json.dumps(return_json))


# detail view
def detail(request, movie_id):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404
    movies = get_object_or_404(Movie, id=movie_id)

    # Get Book Detail
    has_detail = False
    ISBN = movies.ISBN
    detail_book_data = Book_Data.objects.filter(ISBN__icontains=ISBN).distinct()

    author = year = publisher = code = borrow_cnt = ""

    if len(detail_book_data) >= 0:
        book = detail_book_data[0]
        author = book.author
        if author[-1]==',' or author[-1]=='，':
            author = author[:-1]
        year = book.year
        publisher = book.publisher
        borrow_cnt = str(book.borrow_cnt)
        code = book.code
        has_detail = True
        
    # Get ratings (Not Well Implemented)
    ratings = Movie.objects.get(id=movie_id)
    # print(ratings.myrating_set.all())
    rate_cnt = 0
    total_rate = 0.0
    for rate_object in ratings.myrating_set.all():
        total_rate += float(rate_object.rating)
        rate_cnt += 1

    if rate_cnt:
        total_rate /= rate_cnt
    else:
        rate_cnt = None

    # for rating
    if request.method == "POST":
        rate = request.POST['rating']
        # Check Exist Ratings
        user_rating = Myrating.objects.filter(Q(user=request.user)&Q(movie=movies)).distinct()
        if len(user_rating)!=0:
            messages.error(request, "不能重复提交评分！")
        else:
            ratingObject = Myrating()
            ratingObject.user = request.user
            ratingObject.movie = movies
            ratingObject.rating = rate
            ratingObject.save()
            messages.success(request, "您的评分已提交！")
    return render(request, 'web/detail.html', {'movies': movies, 'rate': str(total_rate)[:3], 'rate_cnt': rate_cnt, 'author':author, 'year':year, 'publisher':publisher, 'borrow_cnt':borrow_cnt, 'code':code, 'has_detail':has_detail})


# (App) app_book_detail
def app_book_detail(request):
    ISBN = request.GET.get("test")
    print(ISBN)

    # Get Book Detail
    has_detail = False
    detail_book_data = Book_Data.objects.filter(ISBN__icontains=ISBN).distinct()

    name = author = year = publisher = code = borrow_cnt = ""

    if len(detail_book_data) >= 0:
        book = detail_book_data[0]
        name = book.name
        author = book.author
        if author[-1]==',' or author[-1]=='，':
            author = author[:-1]
        year = book.year
        publisher = book.publisher
        borrow_cnt = str(book.borrow_cnt)
        code = book.code
        has_detail = True
        
    return_json = {
        "name": name,
        "author": author,
        "year": year,
        "publisher": publisher,
        "borrow_cnt": borrow_cnt,
        "code": code
    }

    return HttpResponse(json.dumps(return_json))


# Register user
def signUp(request):
    form = UserForm(request.POST or None)
    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']

        user.set_password(password)
        user.save()
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("index")

    for i, field in enumerate(form):
        if i == 0:
            field.name_tag = '学号'
        else:
            field.name_tag = '密码'

    context = {
        'form': form,
    }
    return render(request, 'web/signUp.html', context)


# Login User
def Login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("recommend")
            else:
                return render(request, 'web/login.html', {'error_message': '您的账号已被停用'})
        else:
            return render(request, 'web/login.html', {'error_message': '无效的学号或密码'})
    return render(request, 'web/login.html')


# (App) app_login
def app_login(request):
    user_id = request.GET.get('UserID')
    password = request.GET.get('Password')

    user = authenticate(username=user_id, password=password)
    if user is not None:
        return_json = {
            "result": 1,
            "token": create_token(user_id=user_id)
        }
        return HttpResponse(json.dumps(return_json))
    else:
        return_json = {
            "result": 0
        }
        return HttpResponse(json.dumps(return_json))

# Logout user
def Logout(request):
    logout(request)
    return redirect("login")
