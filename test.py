import cv2
import face_recognition
import os
import numpy as np

# Directory paths
KNOWN_FACES_DIR = "/home/marof/PycharmProjects/faceRecognition/Student/images"
DETECTED_FACES_DIR = "/home/marof/PycharmProjects/faceRecognition/Student/detect"

# Load known faces
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0]

            # Load the known face image and encode it
            known_image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(known_image)[0]

            known_face_encodings.append(encoding)
            known_face_names.append(name)

    return known_face_encodings, known_face_names

# Save known face
def save_known_face(name, face_encoding):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    np.save(os.path.join(KNOWN_FACES_DIR, f"{name}_encoding.npy"), face_encoding)

# Load detected faces
def load_detected_faces():
    detected_faces = []

    for filename in os.listdir(DETECTED_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(DETECTED_FACES_DIR, filename)
            detected_faces.append(path)

    return detected_faces

# Save detected face
def save_detected_face(frame, face_locations):
    os.makedirs(DETECTED_FACES_DIR, exist_ok=True)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Extract the face from the frame
        face_frame = frame[top:bottom, left:right]

        # Save the detected face
        cv2.imwrite(os.path.join(DETECTED_FACES_DIR, f"detected_face_{i}.jpg"), face_frame)

# Recognize faces in the image
def recognize_faces(image_path):
    known_face_encodings, known_face_names = load_known_faces()

    # Load the image and find face locations
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Iterate through each detected face
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle and display the name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the result
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# Save a known face
# known_image = face_recognition.load_image_file("known_faces/john.jpg")
# known_encoding = face_recognition.face_encodings(known_image)[0]
# save_known_face("John", known_encoding)

# Recognize faces in the image
recognize_faces("trump.jpeg")
