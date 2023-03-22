import cv2
import mediapipe as mp

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
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
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
        print('hand_landmarks:', hand_landmarks)
        print(
        f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # darle la vuelta y retornar
    return cv2.flip(annotated_image, 1)

def draw_text(
        self,
        frame,
        font=cv2.FONT_HERSHEY_COMPLEX,
        font_size=1,
        font_thickness=2,
        offset=int(HEIGHT * 0.02),
        bg_color=(245, 242, 176, 0.85),
    ):
        window_w = int(HEIGHT * len(frame[0]) / len(frame))

        (text_w, text_h), _ = cv2.getTextSize(
            self.sign_detected, font, font_size, font_thickness
        )

        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset

        cv2.rectangle(frame, (0, text_y - offset), (window_w, HEIGHT), bg_color, -1)
        cv2.putText(
            frame,
            self.sign_detected,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (118, 62, 37),
            font_thickness,
        )
        return frame
# Guarde el video si el usuario lo elige
def vid_save():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()