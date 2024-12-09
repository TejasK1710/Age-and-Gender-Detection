import cv2
import numpy as np

# Paths to model files
age_model = r"C:\Users\hp\Desktop\PHN WORK\Projects\age and gender detection\age_net.caffemodel"
age_proto = r"C:\Users\hp\Desktop\PHN WORK\Projects\age and gender detection\deploy_age2 (1).prototxt"
gender_model = r"C:\Users\hp\Desktop\PHN WORK\Projects\age and gender detection\gender_net.caffemodel"
gender_proto = r"C:\Users\hp\Desktop\PHN WORK\Projects\age and gender detection\deploy_gender2.prototxt"

# Load models
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Define labels
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_labels[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_ranges[age_preds[0].argmax()]

        # Draw bounding box and predictions
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Age and Gender Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
