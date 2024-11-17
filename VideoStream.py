#!/usr/bin/env python
import cv2
from threading import Thread 

class VideoStream:
    """
    separate thread for capturing frames from camera to increase frame rate. 
    VideoCapture and read are blocking methods 
    
    TODO add getter instead of accessing variables directly
    add in main thread optional argument to change input camera 
    set default to 0 
    """
    def __init__(self, src = 0):  #optional arg for camera source 
        #start using camera param 0 for index of cam(if multiple)
        self.capture = cv2.VideoCapture(src) 
        #only for initialisation in class. otherwise done in update() 
        (self.ret,self.frame) = self.capture.read()  # returned stream boolean and captured frame 
        
        self.stopped = False
    
    def start(self):
        #daemon terminates automatically after main thread finishes
        #used for indefinite tasks like here
        Thread(target=self.update, daemon=True, args=()).start() 
        return self

    def update(self):
        #run until thread is stopped
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                (self.ret, self.frame) = self.capture.read()
                
                #returns only 1 frame, must be in loop  
                #cv2.imshow("Video",cv2.flip(self.frame,1))
                #cv2.waitKey(0) 
                #if cv2.waitKey(100) == ord('q'):
                #self.stop()
        
    def stop(self):
        # resolved terminate called without active exception        causing core dump 
        self.stopped = True
        self.capture.release()  # release camera resources
        cv2.destroyAllWindows() # close window
        
