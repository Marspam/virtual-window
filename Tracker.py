#!/usr/bin/env python
import cv2
import mediapipe as mp
from threading import Thread
import math
from KalmanFilter import KalmanFilter 

h_fov_rad= (66*math.pi)/180 
v_fov= 50
cam_res= (640, 480)
cam_offset= 11
cam_sensor_w = 4.8 #cm
#focal_length = cam_sensor_w/(2*math.tan(h_fov_rad/2))

"""
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles

get face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode = False,
                max_num_faces = 1,
                min_detection_confidence = 0.8,  #increase confidence to smooth out mesh
                min_tracking_confidence = 0.8
         )
"""
#get face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp.solutions.face_detection.FaceDetection (
                model_selection = 1,
                min_detection_confidence = 0.8 )
 
first = True

class Tracker:
    def __init__(self, frame=None):
        self.frame = frame
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference
        self.frame.flags.writeable = False
        #convert frame to rgb
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)        
        self.stopped = False
        self.nx=0
        self.ny=0        
        self.z = 0
    def start(self):
        Thread(target=self.running, daemon=True, args=()).start() 
        return self

    def running(self):
        #run until thread is stopped
        first = True
        while not self.stopped:  
            #process frame to get list of face landmarks 
            results = face_detection.process(self.frame)    # or face_mesh.process(self.frame)           
            self.frame.flags.writeable = True
            if results.detections:                                             # results.multi_face_landmarks:
                for detection in results.detections:                                                         #face_landmarks in results.multi_face_landmarks:
                    
                    
                    h, w, _ = self.frame.shape # get image dimensions in pixels excluding channel 
                    """
                    mp_drawing.draw_landmarks(
                        
                        image = self.frame,
                        landmark_list= face_landmarks,
                        connections = mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    mp_drawing.draw_landmarks(
                        image = self.frame,
                        landmark_list= face_landmarks,
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                     
                    #using average landmarks doesnt work
                    r_eye_marks = [33,133,159,362,263]
                    l_eye_marks = [362,263,386,374,33,133]
                    r_eye_x=[]
                    r_eye_y=[]
                    l_eye_x=[]
                    l_eye_y=[]
                    for landmark in r_eye_marks:
                        r_eye_x.append(face_landmarks.landmark[landmark].x * w)
                        r_eye_y.append(face_landmarks.landmark[landmark].y * h)
                    for landmark in l_eye_marks:
                        l_eye_x.append(face_landmarks.landmark[landmark].x * w)
                        l_eye_y.append(face_landmarks.landmark[landmark].y * h)
                    r_eye_avg_x = int(np.mean(r_eye_x))  
                    r_eye_avg_y = int(np.mean(r_eye_y))
                    l_eye_avg_x = int(np.mean(l_eye_x))
                    l_eye_avg_y = int(np.mean(l_eye_y))
                    """
                    
                    """
                    #using one eyelid for facemesh
                    r_eyelid_up = face_landmarks.landmark[159] 
                    l_eyelid_up = face_landmarks.landmark[386]
                    nose_tip = face_landmarks.landmark[1]
                    
                    r_eyelid_up_x,r_eyelid_up_y = int(r_eyelid_up.x * w), int(r_eyelid_up.y * h)   
                    l_eyelid_up_x,l_eyelid_up_y=int(l_eyelid_up.x * w), int(l_eyelid_up.y * h)
                    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h) #x and y are normalized 
                    """
                    
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
                    
                    self.nx = predicted_nose_tip_x 
                    self.ny = predicted_nose_tip_y
                    self.z = d
                    """
                    #non predicted
                    cv2.circle(self.frame, (eye_l_x,eye_l_y), 5 ,(0,255,0) , -1) #bgr  
                    cv2.circle(self.frame, (eye_r_x,eye_r_y), 5 ,(0,0,255) , -1)
                    cv2.circle(self.frame, (nose_tip_x, nose_tip_y), 5 , (255, 150, 0), -1)  #5 pixels, blue, -1 for fill 
                   
                    self.nx = nose_tip_x 
                    self.ny = nose_tip_y
                    self.z = d
                    """
                #todo make method that returns images for image show
                #cv2.imshow("Video",cv2.flip(self.frame,1))
                #cv2.waitKey(1) # 1 is non blocking waits 1ms 
                 

    def get_head_coords(self):
        return self.nx,self.ny,self.z
    
   
        
    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()
        

