# 图书推荐系统
## Docker
Install docker, clone this repo and run this:
```
docker build -t python:movie .
```
Then the image is built.
```
docker run -p 6777:8000 --name bookrec -d python:movie
```
where 6777 is the port on your host.

If you want to enter bash, then:
```
docker exec -it bookrec bash
```
