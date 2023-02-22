from django.shortcuts import render
from .forms import UploadForm
from requests import post
import base64
import json
BASE_URL = 'http://127.0.0.1:8000/'


def index(request):
    ctx = {
        'form': UploadForm()
    }

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['img']
            enc = base64.b64encode(file.read())
            ctx['img'] = str(enc.decode('utf8'))
            r = post(BASE_URL+'api/upload/', data={'img': enc})
            data = json.loads(r.text)['message']

            ctx['classes'] = [x['class'] for x in data]
            ctx['confidences'] = [x['confidence']*100 for x in data]

    return render(request, 'index.html', context=ctx)
