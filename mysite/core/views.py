import os
# ignore lack of gpu for keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from .forms import  ImageForm

import urllib
import numpy as np
from script.hand_image_detector import hand_detection
import cv2

from mysite.camera import VideoCamera, gen
from mysite.webcam_manager import *
from django.http import StreamingHttpResponse
import joblib


class Home(TemplateView):
    template_name = 'home.html'


def image_upload_view(request):
    """PROCESA LA IMAGEN SUBIDA POR EL USUARIO"""
    data = {"success": False}
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if request.FILES.get("image", None) is not None:
            image = _grab_image(stream=request.FILES["image"])
            # LLAMA A LA DETECCION
            annotated_image = hand_detection(image)
            # SE MUESTRA LA IMAGEN DE SALIDA
            cv2.imshow("output", annotated_image)
            cv2.waitKey(0)

            form.save()
            img_obj = form.instance
            return render(request, 'image_upload.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
    return render(request, 'image_upload.html', {'form': form})


# una función auxiliar para convertir img.url en un objeto cv.img
# solo para carga y detección de imágenes
def _grab_image(path=None, stream=None, url=None):
	if path is not None:
		image = cv2.imread(path)
	else:	
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
		elif stream is not None:
			data = stream.read()
		# Se convierte la imagen en una matriz NumPy y luego se lee en
		# formato de OpenCV
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# devuelve la imagen
	return image

# for video input and detection
# the whole thing, video
# is returned as a streaming http response, or bytes
def video_stream(request):
    vid = StreamingHttpResponse(gen(VideoCamera(), False), 
    content_type='multipart/x-mixed-replace; boundary=frame')
    return vid

def video_save(request):
    vid = StreamingHttpResponse(gen(WebcamManager(), True), 
    content_type='multipart/x-mixed-replace; boundary=frame')
    return vid

def video_input(request):
    return render(request, 'video_input.html')