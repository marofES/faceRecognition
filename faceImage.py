import cv2
import numpy as np
import face_recognition
import os

image = "/student/images/puyin.jpg"
# Load the pre-trained deep learning model for face detection
net = cv2.dnn.readNetFromCaffe("/home/marof/PycharmProjects/faceRecognition/model/deploy.prototxt", "/home/marof/PycharmProjects/faceRecognition/model/res10_300x300_ssd_iter_140000.caffemodel")

# Load student images and encode them
student_images_folder = "/home/marof/PycharmProjects/faceRecognition/Student/images"
student_image_files = [f for f in os.listdir(student_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
known_face_encodings = []
known_face_names = []

for image_file in student_image_files:
    image_path = os.path.join(student_images_folder, image_file)
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(image_file)[0])

# Load the test image
test_image_path = "nora.jpg"
test_image = cv2.imread(test_image_path)

# Resize the test image for faster processing
small_test_image = cv2.resize(test_image, (0, 0), fx=0.5, fy=0.5)

# Convert the test image to a blob for face detection
blob = cv2.dnn.blobFromImage(small_test_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network to detect faces
net.setInput(blob)
detections = net.forward()


# Process each face detected in the test image
for j in range(detections.shape[2]):
    confidence = detections[0, 0, j, 2]
    print('confidence ', confidence)

    # Filter out weak detections
    if confidence > 0.5:
        # Scale the bounding box to the original test image size
        box = detections[0, 0, j, 3:7] * np.array([2, 2, 2, 2])
        (startX, startY, endX, endY) = box.astype("int")

        # Extract the face from the test image
        face_frame = test_image[startY:endY, startX:endX]

        # Encode the face
        face_encoding = face_recognition.face_encodings(face_frame)

        if len(face_encoding) > 0:
            # Check if the face matches any known student
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0], tolerance=0.5)
            print('matches ',matches)
            name = "Unknown"

            # If a match is found, use the name of the known student
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(name)

            # Draw the bounding box and name on the test image
            cv2.rectangle(test_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(test_image, name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Save the output image
# output_image_path = "image_with_recognition.jpg"
# cv2.imwrite(output_image_path, test_image)
#
# # Display the output image
# cv2.imshow('Output Image', test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
