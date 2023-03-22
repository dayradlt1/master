import cv2
import mediapipe as mp
import numpy as np
from mysite.Funciones.normalizacionCords import obtenerAngulos

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
HEIGHT=600

# este método se usa para deducir el método hand_video
def hand_image():
    # Para imágenes estaticas
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    # Se carga el video
    videoFile = "test_vid.mp4"
    cap = cv2.VideoCapture(videoFile)
    flag, frame = cap.read()

    # mientras cap.isOpened():
    while flag:
        image = cv2.flip(frame, 1)
        frame_ID = cap.get(1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_hight, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print(
                f'meñique: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_hight})'
            )
            print(
                f'anular: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_hight})'
            )
            print(
                f'medio: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_hight})'
            )
            print(
                f'indice: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_hight})'
            )
            print(
                f'pulgar: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_hight})'
            )
            print(
                f'pulgar_interno: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_hight})'
            )

        mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(
            '/tmp/annotated_image_' + str(frame_ID) + '.png', cv2.flip(annotated_image, 1))
        flag, frame = cap.read()
    hands.close()

def hand_video(flag, frame):
    # Para las imagenes estaticas
    # parametros para el detector
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)
    # voltearlo a lo largo del eje y
    image = cv2.flip(frame, 1)
    # Conversion del formato de color
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        hands.close()
        return frame
    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()
    # dibujar resultados de puntos de referencia
    for hand_landmarks in results.multi_hand_landmarks:

        print(
            f'meñique: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_hight})'
        )
        print(
            f'anular: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_hight})'
        )
        print(
            f'medio: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_hight})'
        )
        print(
            f'indice: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_hight})'
        )
        print(
            f'pulgar: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_hight})'
        )
        print(
            f'pulgar_interno: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
    # darle la vuelta y retornar
    return cv2.flip(annotated_image, 1)