# Copyright 2021 Lu
import time
from datetime import datetime
import cv2
import os
import face_recognition
import numpy as np
import requests
import json

path = 'Households'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Database info,
URL = "https://milestone2-ad947-default-rtdb.firebaseio.com/"

# For anonymous sign in
AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyDVsrk7uwW6lLuWm_uuLsCOARFN4eUJTQY";
headers = {'Content-type': 'application/json'}
auth_req_params = {"returnSecureToken": "true"}

# Start connection to Firebase and get anonymous authentication
connection = requests.Session()
connection.headers.update(headers)
auth_request = connection.post(url=AUTH_URL, params=auth_req_params)
auth_info = auth_request.json()
auth_params = {'auth': auth_info["idToken"]}

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


print('Wait for Encoding...')
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Access webcam from PyCharm
print('Opening Camera...')
cap = cv2.VideoCapture(0)

# FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
if not cap.isOpened():
    raise IOError("Cannot open the camera")
count_unknown = 0
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        # find min dis
        matchIndex = np.argmin(faceDis)
        # # find out who
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        name = "Unknown"
        isUnknown = True
        if matches[matchIndex] and faceDis[matchIndex] <= 0.5:
            name = classNames[matchIndex].upper()
            # print('Household ' + name + '. The door is unlocked, welcome home!')
            isUnknown = False
            count_unknown = 0
        else:
            if count_unknown > 2:
                name = 'Unknown'
                # print('Unknown visitor is at the door. ')
                isUnknown = True
            count_unknown += 1
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
        markAttendance(name)
        recognition_results = {
            'name': name,
            'isUnknown': isUnknown,
        }
        detection_results = {
            'faceDis': faceDis[matchIndex],
        }
        # Jasonify the results before sending
        recognition_data_json = json.dumps(recognition_results)
        detection_data_json = json.dumps(detection_results)
        # The URL for the part of the database we will put the detection results
        recognition_url = URL + "recognition.json"
        detection_url = URL + "detection.json"
        # Post the data to the database
        post_recognition_request = connection.put(url=recognition_url, data=recognition_data_json, params=auth_params)
        post_detection_request = connection.put(url=detection_url, data=detection_data_json, params=auth_params)
        # Make sure data is successfully sent
        print("Detection data sent: " + str(post_detection_request.ok))
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
