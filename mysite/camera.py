import cv2
from script.hand_video_detector import hand_video
import time

# toma la imagen de la camara y la convierte a objeto de opencv
# luego es procesado por gen()
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		
	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		if success:
			# llama a la detección aquí
			image = hand_video(success, image)

		return image


# generador que guarda el video capturado si se establece la bandera
def gen(camera, flag):
	if flag == True:
		# time information
		time_now = time.localtime()
		current_time = time.strftime("%H:%M:%S", time_now)
		# default format
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# salida que es un escritor de cv, dado el nombre y el formato, y la resolución
		out = cv2.VideoWriter('output_' + str(current_time) + '.avi',fourcc, 20.0, (640,480))

		while True:
			# objeto cv a jpg
			ret, jpeg = cv2.imencode('.jpg', camera.get_frame())
			# jpg a bytes
			frame =  jpeg.tobytes()
			# Generador cediendo los bytes
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

			cv_frame = camera.get_frame()
			out.write(cv_frame)
	
	else:
		while True:
			ret, jpeg = cv2.imencode('.jpg', camera.get_frame())
			frame =  jpeg.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
