import cv2
from cv2 import CascadeClassifier

def face_detection(path):
    # Create the haar cascade
    faceCascade = CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return len(faces),image     

# if __name__ == '__main__':
#     path='screenshots\\faces.jpeg'
#     num,image=face_detection(path)
#     print("detect: "+str(num)+" faces")
#     cv2.imshow("Faces found", image)
#     cv2.waitKey(0)
