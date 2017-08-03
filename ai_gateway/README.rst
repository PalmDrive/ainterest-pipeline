Usage

启动server

```
virtualenv env
source env/bin/activate

pip install -r requirements.txt
# 如果install ailab 失败 run: pip install --upgrade "git+ssh://git@github.com/PalmDrive/ainterest-pipeline.git#egg=ailab&subdirectory=ailab"

python manage.py runserver 8080

```

client请求


```
//其中 type是分类类型, content 是文章string
curl -H "Content-Type: application/json" -X GET -d '{"type":"field","content":"xyz"}' localhost:8080/api/classify
```
