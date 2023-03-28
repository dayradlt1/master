import cv2
import mediapipe as mp
from mysite.Funciones.condicionales import condicionalesLetras
from mysite.Funciones.normalizacionCords import obtenerAngulos



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


# toma la imagen de la camara y la convierte a objeto de opencv
# luego es procesado por gen()
class VideoCamera(object):
        def __init__(self):
                self.video = cv2.VideoCapture(0)
                wCam, hCam = 1280, 720
                self.video.set(3, wCam)
                self.video.set(4, hCam)
        def __del__(self):
                self.video.release()

        def get_video(self):
                lectura_actual = 0
                with mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.75) as hands:

                    while True:
                        var_b, video = self.video.read() #read desempaqueta una vari booleana y la imagen
                        if var_b == False:
                            break
                        height, width, _ = video.shape
                        video = cv2.flip(video, 1)#flip voltea una matriz 2d en este caso video
                        video_rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)#convierte la imagen de un espacio de color a otro
                        results = hands.process(video_rgb)
                        if results.multi_hand_landmarks is not None:
                            # Accediendo a los puntos de referencia, de acuerdo a su nombre
                                
                                angulosid = obtenerAngulos(results, width, height)[0]

                                dedos = []
                                # pulgar externo angle
                                if angulosid[5] > 125:
                                    dedos.append(1)
                                else:
                                    dedos.append(0)

                                # pulgar interno
                                if angulosid[4] > 150:
                                    dedos.append(1)
                                else:
                                    dedos.append(0)

                                # 4 dedos
                                for id in range(0, 4):
                                    if angulosid[id] > 90:
                                        dedos.append(1)
                                    else:
                                        dedos.append(0)

                                
                                TotalDedos = dedos.count(1)
                                condicionalesLetras(dedos, video)
                                
                                pinky = obtenerAngulos(results, width, height)[1]
                                pinkY=pinky[1] + pinky[0]
                                resta = pinkY - lectura_actual
                                lectura_actual = pinkY 
                                print(abs(resta), pinkY, lectura_actual)
                                
                                if dedos == [0, 0, 1, 0, 0, 0]:
                                    if abs(resta) > 30:
                                        print("jota en movimento")
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.rectangle(video, (0, 0), (100, 100), (255, 255, 255), -1)
                                        cv2.putText(video, 'J', (20, 80), font, 3, (0, 0, 0), 2, cv2.LINE_AA)
                                        

                                if results.multi_hand_landmarks:
                                    for hand_landmarks in results.multi_hand_landmarks:
                                        mp_drawing.draw_landmarks(
                                            video,
                                            hand_landmarks,
                                            mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())
                        return video
                        #if cv2.waitKey(1) & 0xFF == 27:
                            #break