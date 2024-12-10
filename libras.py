import cv2
import mediapipe as mp

# Inicializar captura de vídeo e soluções do Mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)

# IDs das pontas dos dedos conforme Mediapipe
finger_tips = [4, 8, 12, 16, 20]
mid_joints = [3, 7, 11, 15, 19]

# Função para identificar gestos com base no alfabeto de Libras


def detect_libras_gestures(image, hand_landmarks, no=0):
    if hand_landmarks:
        landmarks = hand_landmarks[no].landmark

        # Listas para armazenar estados e direções dos dedos
        fingers_up = []
        fingers_down = []

        for tip in finger_tips:
            fingertop_y = landmarks[tip].y
            fingerbottom_y = landmarks[tip - 2].y
            fingertop_x = landmarks[tip].x
            fingerbottom_x = landmarks[tip - 2].x

            # Verificar se o dedo está levantado ou abaixado
            if tip == 4:  # Polegar (movimento lateral no eixo X)
                fingers_up.append(fingertop_x > fingerbottom_x)
                fingers_down.append(False)
            else:  # Outros dedos (movimento vertical no eixo Y)
                fingers_up.append(fingertop_y < fingerbottom_y)
                fingers_down.append(fingertop_y > fingerbottom_y)

        # Verificar curvatura dos dedos para o gesto "C"
        is_curved = []
        for tip, mid in zip(finger_tips, mid_joints):
            fingertop_x = landmarks[tip].x
            fingertop_y = landmarks[tip].y
            mid_joint_x = landmarks[mid].x
            mid_joint_y = landmarks[mid].y

            # Um dedo é considerado curvado se a ponta estiver próxima à articulação média
            distance_x = abs(fingertop_x - mid_joint_x)
            distance_y = abs(fingertop_y - mid_joint_y)
            is_curved.append(distance_x < 0.05 and distance_y < 0.05)

        # Lógica para identificar gestos específicos
        gesture = ""
        # Indicador e médio dobrados para baixo
        if fingers_down == [False, True, True, False, False]:
            gesture = "N"
        elif fingers_up == [True, False, False, False, False]:  # Apenas polegar levantado
            gesture = "A"
        elif all(is_curved):  # Todos os dedos curvados para formar "C"
            gesture = "C"

        # Mostrar o gesto na tela
        if gesture:
            cv2.putText(image, gesture, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Função para desenhar os marcos das mãos


def draw_hand_landmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(
                image, landmarks, mp_hands.HAND_CONNECTIONS)


# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    result = hands.process(frame)

    hand_landmarks = result.multi_hand_landmarks
    draw_hand_landmarks(frame, hand_landmarks)
    detect_libras_gestures(frame, hand_landmarks)

    cv2.imshow("Detector de Mãos - Libras", frame)

    if cv2.waitKey(1) & 0xFF == 32:  # Pressione espaço para sair
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
