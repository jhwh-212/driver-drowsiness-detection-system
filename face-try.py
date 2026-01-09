# -- coding: utf-8 --
import os
import cv2
import sys

# Load Haar cascade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(BASE_DIR, 'models', 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"[ERROR] Failed to load cascade from {cascade_path}")
    sys.exit(1)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    sys.exit(1)

try:
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            continue

        # Detection on REAL frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Mirror ONLY for display
        display = cv2.flip(img, 1)
        cv2.imshow('Face Detection', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()