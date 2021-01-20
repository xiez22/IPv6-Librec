from django.contrib import admin
from .models import Movie,Myrating,Book_Data,User_book

admin.site.register(Movie)
admin.site.register(Myrating)
admin.site.register(Book_Data)
admin.site.register(User_book)
# Register your models here.
