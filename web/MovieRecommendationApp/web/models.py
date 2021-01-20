from django.contrib.auth.models import Permission, User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

class Movie(models.Model):
	title   	= models.CharField(max_length=200)
	genre  		= models.CharField(max_length=100)
	movie_logo  = models.FileField()
	ISBN        = models.CharField(max_length=100)  

	def __str__(self):
		return self.title

class Myrating(models.Model):
	user   	= models.ForeignKey(User,on_delete=models.CASCADE) 
	movie 	= models.ForeignKey(Movie,on_delete=models.CASCADE)
	rating 	= models.IntegerField(default=1,validators=[MaxValueValidator(5),MinValueValidator(0)])

class User_book(models.Model):
	# user    = models.ForeignKey(User,on_delete=models.CASCADE)
	student_id = models.CharField(max_length=100)
	book   = models.CharField(max_length=100)

class Book_Data(models.Model):
	book_id = models.IntegerField()
	code = models.TextField()
	name = models.TextField()
	author = models.TextField()
	ISBN = models.TextField()
	publisher = models.TextField()
	year = models.TextField()
	borrow_cnt = models.IntegerField()
	rate_cnt = models.IntegerField()
	total_rate = models.TextField()
