# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json

from django.http import HttpResponse


def classify(request):
    data = json.loads(request.body)
    print data
    return HttpResponse("Hello, world. You're at the polls index.")
