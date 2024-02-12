# CS_Artificial_Intelligence
pip install opencv-python
import cv2

# Load pre-trained classifiers
pedestrian_cascade = cv2.CascadeClassifier('haarcascade_fullbody.XML')
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Function to detect pedestrians and vehicles
def detect_objects(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

# Capture video
cap = cv2.VideoCapture('i0yqhHKWY0A')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect pedestrians and vehicles
    frame = detect_objects(frame)
    
    # Display the result
    cv2.imshow('Pedestrian and Vehicle Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
