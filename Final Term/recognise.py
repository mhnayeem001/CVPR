import cv2
import xlwrite
import time

start = time.time()
period = 30  # Keep the pop-up window open for 10 seconds
face_cas = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train/train.yml')
flag = 0
id = 0
filename = 'filename'
dict = {}

# Font for text
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)

    if len(faces) == 0:
        # No faces detected, display "Unknown"
        cv2.putText(img, "Unknown", (50, 50), font, 1, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(roi_gray)

        if conf > 50:
            if id == 1:
                id = 'Tanzim'
                if id not in dict:
                    filename = xlwrite.output('attendance', 'class1', 1, id, 'yes')
                    dict[id] = id

            elif id == 2:
                id = 'Nayeem'
                if id not in dict:
                    filename = xlwrite.output('attendance', 'class1', 2, id, 'yes')
                    dict[id] = id

            elif id == 3:
                id = 'Raiyan'
                if id not in dict:
                    filename = xlwrite.output('attendance', 'class1', 3, id, 'yes')
                    dict[id] = id

            elif id == 4:
                id = 'Alvi'
                if id not in dict:
                    filename = xlwrite.output('attendance', 'class1', 4, id, 'yes')
                    dict[id] = id
        else:
            id = 'Unknown'
            flag += 1

        # Display the ID and confidence
        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)

    # Show the frame
    cv2.imshow('frame', img)

    # Termination conditions
    if time.time() > start + period:  # Ensure it runs for 10 seconds
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):  # Allow manual termination with 'q'
        break

cap.release()
cv2.destroyAllWindows()
