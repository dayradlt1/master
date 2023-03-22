import cv2
import mediapipe as mp
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
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
      )
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    output_relative_path = '/tmp/annotated_image/' + str(idx) + '.png'
    cv2.imwrite(output_relative_path, cv2.flip(annotated_image, 1))
  hands.close()

  return cv2.flip(annotated_image, 1)