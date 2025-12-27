import cv2
import pickle
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

mdl_dict = pickle.load(open('./model.p', 'rb'))
model = mdl_dict['model']

hands_img = mp.solutions.hands
draw_img = mp.solutions.drawing_utils
draw_img_style = mp.solutions.drawing_styles

hands = hands_img.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataset_labels = {0: 'I', 1: 'Love', 2: 'You', 3: 'Eldow', 4: 'FAI'}

stop_key = ord('q')

while True:
    data_aux = []
    x_coordinates = []
    y_coordinates = []

    ret, frame = cap.read()

    height, width, _ = frame.shape

    converted_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_landmark_results = hands.process(converted_rgb_frame)
    if hand_landmark_results.multi_hand_landmarks:
        for hand_landmarks in hand_landmark_results.multi_hand_landmarks:
            draw_img.draw_landmarks(
                frame,
                hand_landmarks,
                hands_img.HAND_CONNECTIONS,
                draw_img_style.get_default_hand_landmarks_style(),
                draw_img_style.get_default_hand_connections_style())

        for hand_landmarks in hand_landmark_results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_coordinates.append(x)
                y_coordinates.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_coordinates))
                data_aux.append(y - min(y_coordinates))

        while len(data_aux) < 42:
            data_aux.append(0)

        min_x = min(x_coordinates)
        min_y = min(y_coordinates)
        max_x = max(x_coordinates)
        max_y = max(y_coordinates)

        frame_size = 30
        x1 = max(0, int(min_x * width) - frame_size)
        y1 = max(0, int(min_y * height) - frame_size)
        x2 = min(width, int(max_x * width) + frame_size)
        y2 = min(height, int(max_y * height) + frame_size)

        while len(data_aux) > 42:
            data_aux.pop()

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = dataset_labels[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        cv2.putText(frame, "Press 'Q' to stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == stop_key:
        break

cap.release()
cv2.destroyAllWindows()
