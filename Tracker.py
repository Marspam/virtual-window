#!/usr/bin/env python
import cv2
import mediapipe as mp
from threading import Thread
import math
import time 
from KalmanFilter import KalmanFilter 

#get face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp.solutions.face_detection.FaceDetection (
                model_selection = 1,
                min_detection_confidence = 0.8 )
 
first = True
class Tracker:
    """ 
    This class calculates the face coordinates and depth   
    added setter to enable face tracking window
    """

    def __init__(self, frame=None):
        self.frame = frame
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference
        self.frame.flags.writeable = False
        #convert frame to rgb
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)        
        self.stopped = False
        self.show_frames = False
        self.nx=0
        self.ny=0        
        self.z = 0
    
    def start(self):
        Thread(target=self.running, daemon=True, args=()).start() 
        return self

    def running(self):
        global first;
        #run until thread is stopped        
        while not self.stopped:  
            #process frame to get list of face landmarks 
            results = face_detection.process(self.frame)    # or face_mesh.process(self.frame)           
            self.frame.flags.writeable = True
            if results.detections:                                          
                for detection in results.detections:                                                     
                    h, w, _ = self.frame.shape # get image dimensions in pixels excluding channel 

                    eye_l = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                    eye_r = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                    nose_tip = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                    
                    eye_l_x, eye_l_y = int(eye_l.x * w), int(eye_l.y * h) # get pixel coords from normalized
                    eye_r_x, eye_r_y = int(eye_r.x * w), int(eye_r.y * h)
                    nose_tip_x, nose_tip_y = int(nose_tip.x * w), int(nose_tip.y * h)
                  
                    if first: # set values on first run 
                        head_x = nose_tip_x
                        head_x2 = head_x
                        head_x3 = head_x
                        head_y = nose_tip_y
                        head_y2 = head_y
                        head_y3 = head_y
                        first = False
                         
                    #save last 3 values for averaging
                    head_x3 = head_x2
                    head_x2 = head_x
                    head_x = nose_tip_x
                    head_x = (head_x + head_x2 + head_x3) / 3

                    head_y3 = head_y2
                    head_y2 = head_y
                    head_y = nose_tip_y
                    head_y = (head_y + head_y2 + head_y3) / 3
                    
                    #use kalman filter to predict next sample
                    kf = KalmanFilter()
                    predicted_eye_l_x, predicted_eye_l_y = kf.predict(eye_l_x,eye_l_y)
                    predicted_eye_r_x, predicted_eye_r_y = kf.predict(eye_r_x,eye_r_y)
                    predicted_nose_tip_x, predicted_nose_tip_y = kf.predict(head_x, head_y)
                   
                    #geometric estimation
                    #euclidean distance between eyes in pixels
                    e_dist = math.sqrt((predicted_eye_r_x - predicted_eye_l_x)**2 
                                       + (predicted_eye_r_y - predicted_eye_l_y)**2 ) 
                    focal_length = 730        # (e_dist * 45) / 6.3
                    d = (focal_length * 6.3) / e_dist  
                      
                    
                    #draw circle at landmarks
                    cv2.circle(self.frame, (predicted_eye_l_x,predicted_eye_l_y), 5 ,(0,255,0) , -1) #bgr  
                    cv2.circle(self.frame, (predicted_eye_r_x,predicted_eye_r_y), 5 ,(0,0,255) , -1)
                    cv2.circle(self.frame, (predicted_nose_tip_x, predicted_nose_tip_y), 5 , (255, 150, 0), -1)  #5 pixels, blue, -1 for fill          
                    
                    cv2.circle(self.frame, (int(head_x), int(head_y)), 5 , (150, 255, 0), 3)  #5 pixels, blue, -1 for fill          
                    

                    self.nx = predicted_nose_tip_x 
                    self.ny = predicted_nose_tip_y
                    self.z = d 
                      
                if self.show_frames:
                    cv2.imshow("Video",cv2.flip(self.frame,1))
                    cv2.waitKey(1) # 1 is non blocking waits 1ms 
                 

    def get_head_coords(self):
        return self.nx,self.ny,self.z
    
    def set_show_frames(self):
        self.show_frames = True
        
    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()
        

