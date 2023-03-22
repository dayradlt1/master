import cv2
import mediapipe as mp
import numpy as np
import os


def hand_detection(image_path):
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands

  # Para imagenes estaticas
  hands = mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5)
  # Lee una imagen, la gira alrededor del eje y para obtener una salida correcta
  idx = 0
  image = cv2.flip(image_path, 1)
  # Convierte la imagen BGR a RGB antes de procesarla.
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Imprime y dibuja puntos de referencia de la mano en la imagen.
  print('Handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    print("WARNING: This image has no hand(s)!!!")
  else:
    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()
    def get_xy(index):
        return [
            int(hand_landmarks.landmark[index].x * image_width),
            int(hand_landmarks.landmark[index].y * image_hight),
        ]
    for hand_landmarks in results.multi_hand_landmarks:
       # Diccionario de coordenadas de cada dedo
        dedos = {
            "meñique": [
                get_xy(mp_hands.HandLandmark.PINKY_TIP),
                get_xy(mp_hands.HandLandmark.PINKY_PIP),
                get_xy(mp_hands.HandLandmark.PINKY_MCP),
            ],
            "anular": [
                get_xy(mp_hands.HandLandmark.RING_FINGER_TIP),
                get_xy(mp_hands.HandLandmark.RING_FINGER_PIP),
                get_xy(mp_hands.HandLandmark.RING_FINGER_MCP),
            ],
            "medio": [
                get_xy(mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                get_xy(mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                get_xy(mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
            ],
            "indice": [
                get_xy(mp_hands.HandLandmark.INDEX_FINGER_TIP),
                get_xy(mp_hands.HandLandmark.INDEX_FINGER_PIP),
                get_xy(mp_hands.HandLandmark.INDEX_FINGER_MCP),
            ],
            "pulgar": [
                get_xy(mp_hands.HandLandmark.THUMB_TIP),
                get_xy(mp_hands.HandLandmark.THUMB_IP),
                get_xy(mp_hands.HandLandmark.THUMB_MCP),
            ],
            "pulgar_interno": [
                get_xy(mp_hands.HandLandmark.THUMB_TIP),
                get_xy(mp_hands.HandLandmark.THUMB_MCP),
                get_xy(mp_hands.HandLandmark.WRIST),
            ],
        }
        # Extraccion de puntos (x, y)
        dedos_array = np.concatenate([np.array(dedo) for dedo in dedos.values()])
        # El código convierte la lista de puntos en una matriz con forma (6, 3, 2)
        # para poder realizar operaciones vectoriales en bloque.
        puntos = dedos_array.reshape(-1, 3, 2)

        # Calculo de distancia
        l_list = np.linalg.norm(puntos[:, [0, 1], :] - puntos[:, [1, 2], :], axis=2)

        # Calculo 5 angulos
        num_den_list = np.sum(
            (puntos[:, 0, :] - puntos[:, 1, :]) * (puntos[:, 2, :] - puntos[:, 1, :]),
            axis=1,
        ) / (l_list[:, 0] * l_list[:, 1])

        angulos = np.degrees(np.arccos(num_den_list))
        pinky = [
            int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_hight),
        ]
        angulos = angulos.tolist()
        return [angulos, pinky]