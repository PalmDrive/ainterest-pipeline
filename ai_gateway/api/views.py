# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json

from django.http import HttpResponse, JsonResponse

from api import classifiers


def classify(request):
    data = json.loads(request.body)
    classifier = classifiers[data['type']]
    result = classifier.classify(data['content'].encode('utf-8'))
    # result = ['ta', 'adf']
    return JsonResponse({'status': 'ok', 'result': result})
