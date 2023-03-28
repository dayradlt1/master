import os
# ignore lack of gpu for keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.urls import reverse_lazy
from mysite.app import *
from django.views.decorators import gzip
import time


import urllib
import numpy as np
import cv2

from django.http import StreamingHttpResponse

class Home(TemplateView):
    template_name = 'home.html'

def gen(camera,flag):
    if flag == True:
            # time information
            time_now = time.localtime()
            current_time = time.strftime("%H:%M:%S", time_now)
            # default format
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # salida que es un escritor de cv, dado el nombre y el formato, y la resolución
            out = cv2.VideoWriter('output_' + str(current_time) + '.mp4',fourcc, 20.0, (440,280))

            while True:
                # objeto cv a jpg
                ret, jpeg = cv2.imencode('.jpg', camera.get_video())
                # jpg a bytes
                video =  jpeg.tobytes()
                # Generador cediendo los bytes
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + video + b'\r\n\r\n')

                cv_frame = camera.get_video()
                out.write(cv_frame)
        
    else:
            while True:
                ret, jpeg = cv2.imencode('.jpg', camera.get_video())
                video =  jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + video + b'\r\n\r\n')

# for video input and detection
# the whole thing, video
# is returned as a streaming http response, or bytes
@gzip.gzip_page
def video_stream(request):
    vid = StreamingHttpResponse(gen(VideoCamera(),False), 
    content_type='multipart/x-mixed-replace; boundary=frame')
    return vid

def video_input(request):
    return render(request, 'video_input.html')