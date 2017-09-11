# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import sys
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from api import classifiers, similarity

reload(sys)
sys.setdefaultencoding('utf-8')


def classify(request):
    data = json.loads(request.body)
    classifier = classifiers[data['type']]
    result = classifier.classify(data['content'].encode('utf-8'))
    return HttpResponse(
        json.dumps(
                {'status': 'ok', 'result': result},
                ensure_ascii=False,
        ),
        content_type="application/json; encoding=utf-8")


@csrf_exempt
def duplicate(request):
    data = json.loads(request.body)
    sim_articles = similarity.similar(
        data['content'].encode('utf-8'),
        data['id'].encode('utf-8'),
        thres=0.67,
    )
    status = 'ok'
    if len(sim_articles) > 0:
        status = 'error'

    return HttpResponse(
        json.dumps(
                {'status': status, 'duplicate_articles': sim_articles},
                ensure_ascii=False,
        ),
        content_type="application/json; encoding=utf-8"
    )
