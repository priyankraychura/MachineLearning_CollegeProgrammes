# Import required libraries
import cv2

# Read input image
img = cv2.imread('Person.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

# Check if Haar cascades are loaded properly
if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
    print("Error loading Haar cascades!")
    exit()

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print('Number of detected faces:', len(faces))

# Loop over the detected faces
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Draw rectangle around face
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Detect mouth within the lower half of the face
    roi_mouth_gray = gray[y + int(h * 0.6): y + h, x:x + w]
    roi_mouth_color = img[y + int(h * 0.6): y + h, x:x + w]

    mouths = mouth_cascade.detectMultiScale(roi_mouth_gray, 1.5, 10)
    for (mx, my, mw, mh) in mouths:
        cv2.rectangle(roi_mouth_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
        cv2.putText(roi_mouth_color, "Mouth", (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display the final image with detected features
cv2.imshow('Face, Eyes & Mouth Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()