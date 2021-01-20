FROM python
RUN pip install Django==2.2.1 pandas numpy scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./ /MovieRecommendation/

EXPOSE 8000
WORKDIR /MovieRecommendation/MovieRecommendationApp/
ENTRYPOINT ["python","manage.py","runserver","0.0.0.0:8000"] 
