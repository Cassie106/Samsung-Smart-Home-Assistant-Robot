# Copyright 2021 Lu
from datetime import datetime
import cv2
import os
import face_recognition
import numpy as np

path = 'Households'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendances.csv', 'r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            if entry != '.DS_Store':
                nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%D.  %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

print('Encoding Complete')

# Access webcam from PyCharm
cap = cv2.VideoCapture(0)

# FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
if not cap.isOpened():
    raise IOError("Cannot open the camera")

while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        # find out who
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print('Household ' + name + '. The door is unlocked, welcome home!')
            markAttendance(name)
        else:
            name = 'Unknown'
            print('Unknown visitor is at the door. ')
            markAttendance(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            markAttendance(name)

# cv2.imshow('Webcam', img)
# cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()



