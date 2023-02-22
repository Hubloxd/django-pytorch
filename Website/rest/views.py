import io

from base64 import b64decode
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions

from ImageRecognition.main import inference, Model


def handle_upload(f):
    img = Image.open(f)
    return inference(Model, img)


# Create your views here.
class UploadView(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        data: dict[any] = request.data

        if 'img' not in data:
            return Response({"message": "invalid request"}, status=400)
        img = b64decode(data['img'])
        buf = io.BytesIO(img)
        return Response({"message": handle_upload(buf)})
