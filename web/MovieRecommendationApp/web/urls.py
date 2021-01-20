from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:movie_id>/',views.detail ,name='detail'),
    path('signup/',views.signUp,name='signup'),
    path('login/',views.Login,name='login'),
    path('logout/',views.Logout,name='logout'),
    path('recommend/',views.recommend,name='recommend'),
    # For HTML App Service
    path('app_hot_recommend/', views.app_hot_recommend, name='app_hot_recommend'),
    path('app_book_detail/', views.app_book_detail, name='app_book_detail'),
    path('app_login/', views.app_login, name='app_login'),
    path('app_recommend/', views.app_recommend, name='app_recommend')
]
