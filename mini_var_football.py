import cv2
import mediapipe as mp

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Captura de video desde la cámara predeterminada
cap = cv2.VideoCapture(0)

# Variables para el seguimiento de cruce
prev_side = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen para efecto espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    center_x = w // 2

    # Dibujar línea vertical central
    cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 0), 2)

    # Procesamiento de la imagen para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    red_on = False

    if result.multi_hand_landmarks:
        # Tomar la primera mano detectada
        for hand_landmarks in result.multi_hand_landmarks:
            # Obtener coordenada X del índice (landmark 8)
            x_norm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        x_px = int(x_norm * w)

        # Determinar lado actual
        current_side = 'left' if x_px < center_x else 'right'

        if prev_side is None:
            prev_side = current_side
        elif current_side != prev_side:
            red_on = True
            prev_side = current_side

        # Dibujar puntos de referencia de la mano
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Si hay cruce, mostrar foco rojo
    if red_on:
        # Dibujar un círculo rojo en la esquina superior izquierda
        cv2.circle(frame, (50, 50), 30, (0, 0, 255), -1)

    # Mostrar la ventana de video
    cv2.imshow('Hand Crossing Detector', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
