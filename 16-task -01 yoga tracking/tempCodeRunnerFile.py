import cv2
import mediapipe as mp
import numpy as np


mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose


cam=cv2.VideoCapture(0)


def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    if angle>180.0:
        angle=360-angle
    return angle


while True:
    status,image=cam.read()
    if status:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            results=pose.process(image)
            if results.pose_landmarks:

                landmarks=results.pose_landmarks.landmark
                lshoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                lelbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lwrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                lhip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                lankle=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                lknee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                rshoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                relbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rwrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                rhip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rankle=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                rknee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                langle=calculate_angle(lshoulder,lelbow,lwrist)
                rangle=calculate_angle(rshoulder,relbow,rwrist)
                lsangle=calculate_angle(lhip,lshoulder,lelbow)
                rsangle=calculate_angle(rhip,rshoulder,relbow)
                rhangle=calculate_angle(rshoulder,rhip,rknee)
                lhangle=calculate_angle(lshoulder,lhip,lknee)
                rkangle=calculate_angle(rankle,rknee,rhip)
                lkangle=calculate_angle(lankle,lknee,lhip)
                print(langle,rangle,lsangle,rsangle,rhangle,lhangle,rkangle,lkangle)
                if (lsangle>70 and lsangle<100) and (rsangle>80 and rsangle<110):
                    res='T pose'
                else:
                    res=''
                cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
                cv2.putText(image,res,(60,60),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,255),2,cv2.LINE_AA)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                )
                image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                cv2.imshow('result',image)
                cv2.waitKey(1)