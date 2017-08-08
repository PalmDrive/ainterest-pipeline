# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import sys
from django.http import HttpResponse
from api import classifiers

reload(sys)
sys.setdefaultencoding('utf-8')


def classify(request):
    data = json.loads(request.body)
    classifier = classifiers[data['type']]
    result = classifier.classify(data['content'].encode('utf-8'))
    return HttpResponse(
            json.dumps({'status': 'ok', 'result': result}, ensure_ascii=False), content_type="application/json; encoding=utf-8")
