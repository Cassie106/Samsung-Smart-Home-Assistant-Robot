# pip install cmake
# pip install dlib
# pip install numpy
# pip install opencv-python
# pip install face_recognition

import cv2
import face_recognition

# Load Images
imgCass = face_recognition.load_image_file('ImagesBasic/Cass.jpg')
imgCass = cv2.cvtColor(imgCass, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Cass Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('Images/household_3.JPG')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgCass)[0]
encodeCass = face_recognition.face_encodings(imgCass)[0]
cv2.rectangle(imgCass,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]),(255,0,255),2)
# print(faceLoc)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeCass],encodeTest)
faceDis = face_recognition.face_distance([encodeCass],encodeTest)
print(results, faceDis)
# cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

cv2.imshow('Cass Yin', imgCass)
cv2.imshow('Cass Test', imgTest)
cv2.waitKey(0)
