import os
import cv2

TARGET_DIR = './Images'
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

no_of_classification_classes = 5
no_of_samples_per_class = 100

cap = cv2.VideoCapture(0)

for j in range(no_of_classification_classes):
    class_dir = os.path.join(TARGET_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}. Get ready!')

    for _ in range(6):
        ret, frame = cap.read()
        cv2.putText(frame, 'Get ready to perform the new gesture!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

    print('Recording... Press "Q" to stop.')

    collected_samples = 0
    while collected_samples < no_of_samples_per_class:
        ret, frame = cap.read()
        cv2.putText(frame, f'Class: {j} - Recording {collected_samples}/{no_of_samples_per_class}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

        cv2.imwrite(os.path.join(class_dir, f'{collected_samples}.jpg'), frame)
        collected_samples += 1

print('Dataset Sample Collection Complete')

cap.release()
cv2.destroyAllWindows()
