#!/usr/bin/env python 
from VideoStream import VideoStream 
from Tracker import Tracker
from OneEuroFilter import OneEuroFilter
import signal
import os 
import sys
import cv2
import time 
import mediapipe as mp
import numpy as np
import argparse
#disable annoying pygame messages
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

def conv_cv_alpha(cv_image, mask):
    b, g, r = cv2.split(cv_image)
    rgba = [r, g, b, mask]
    cv_image = cv2.merge(rgba,4)
          
    return cv_image

def get_depth_layers(depth_map):
    layers = []
    prev_thres = 255
    div = 30
    for thres in range(255 - div, 0 , -div):
        ret, mask = cv2.threshold(depth_map, thres, 255, cv2.THRESH_BINARY)
        ret, prev_mask = cv2.threshold(depth_map, prev_thres, 255, cv2.THRESH_BINARY)

        prev_thres = thres
        inpaint_img = cv2.inpaint(img, prev_mask, 10, cv2.INPAINT_NS)
        layer = cv2.bitwise_and(inpaint_img, inpaint_img, mask = mask)
        layers.append(conv_cv_alpha(layer, mask))
    #last layer 
    mask = np.zeros(depth_map.shape, np.uint8)
    mask[:,:] = 255
    ret, prev_mask = cv2.threshold(depth_map, prev_thres, 255, cv2.THRESH_BINARY)
    inpaint_img = cv2.inpaint(img, prev_mask, 10 , cv2.INPAINT_NS)
    layer = cv2.bitwise_and(inpaint_img, inpaint_img , mask = mask )
    layers.append(conv_cv_alpha(layer, mask))
    layers = layers[::-1]
 
    return layers
    
def extension_check(param):
    base,ext = os.path.splitext(param)
    if ext.lower() not in ('.jpg', '.png'):
        raise argparse.ArgumentTypeError('file must be a .png or .jpg')
    return param

def argument_handler():
    parser = argparse.ArgumentParser(prog="Virtual window", description="User inputs an image and the program tracks the movement of his face to adjust the image creating a \"Window effect\" ")
    parser.add_argument("-i" , type= extension_check, help="image input", required=True)
    parser.add_argument("-v", help="show tracking output of camera", required=False, action='store_true')
    parser.add_argument("-s", type = int, default = 0, help="Change input camera source index if using multiple camera on system, default is zero", required=False)
    args = parser.parse_args()
    return args

def ema_filter(current_head_z):
    global prev_head_z
    filtered_head_z = alpha * current_head_z + (1 - alpha) * prev_head_z
    prev_head_z = filtered_head_z
    return filtered_head_z

base_z = 35
starttime = time.time()
min_cutoff = 0.004 #cutoff freq
alpha = 0.2 #ema smoothing factor
beta = 0.8 #euro smoothing factor
first_head_z = 0
prev_head_z = 0

def main():  
    try:
        stream = VideoStream(argument_handler().s)
        stream.start()
        tracker = Tracker(stream.frame).start() 
        if argument_handler().v:
            tracker.set_show_frames()
        first = True
        running = True
        while running:
            if stream.stopped or tracker.stopped:
                stream.stop()
                tracker.stop()
                
            frame = stream.frame
            time.sleep(0.001) 
            tracker.frame = frame
            head_coords = tracker.get_head_coords() 
            head_x,head_y,head_z= head_coords[0], head_coords[1],head_coords[2] #nose x, y and distance             
                
            if first:
                first_head_z = head_z
                first = False
            one_euro_filter = OneEuroFilter(starttime, first_head_z , min_cutoff=min_cutoff, beta=beta) 
            filtered_head_z = one_euro_filter(time.time(), head_z )
            filtered_head_z = ema_filter(filtered_head_z)
            
            #head movement 
            #when looking left we want the image to go right
            offset_x = (head_x - (image_rect.width // 2)) * 0.7 
            offset_y = (head_y - (image_rect.height // 2)) * 0.7
             
            #set minimum so it doesnt zoom out more than the image
            #optimal minimum depends on image size
            z_scale_factor = max(0.8,np.round(filtered_head_z/base_z,7))  
    
            scaled_width = int(image_rect.width * z_scale_factor)   
            scaled_height = int(image_rect.height * z_scale_factor)        
            scaled_image = pygame.transform.scale_by(window_image, z_scale_factor)
            scaled_rect = scaled_image.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2)) 
            
            #try parentheses before //2
            image_x = ((image_rect.width - scaled_width) // 2)  + int(offset_x)
            image_y = ((image_rect.height  - scaled_height) // 2) + int(offset_y)
                     
            #width, height = layers[0].get_width, layers[0].get_height()       

            screen.fill((0,0,0)) 
            screen.blit(scaled_image, (int(image_x),int(image_y)))   
            pygame.display.flip()
                 
            #moved last to remove errors after closing pygame when it wants to access methods
            for event in pygame.event.get():
                if event.type == pygame.QUIT: #alt f4 or close button
                    stream.stop()
                    tracker.stop()
                    pygame.quit()
                    sys.exit()
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        stream.stop()
                        tracker.stop()
                        pygame.quit()      
                        running = False
                        sys.exit("ESCAPE")
                    if event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL: #in pygame window
                        stream.stop()
                        tracker.stop()
                        pygame.quit()  
                        running = False
                        sys.exit("INTERRUPT")
    
    except KeyboardInterrupt: #for console 
        stream.stop()
        tracker.stop()
        pygame.quit()
        sys.exit("ABORT\n")

 
if __name__ == "__main__":    
    #inits pygame when we run --help
    pygame.init()
    flags= pygame.FULLSCREEN | pygame.RESIZABLE
    pygame.display.set_caption("Virtual Window")
    screen = pygame.display.set_mode((0,0), flags) #fit native resolution 
    window_image = pygame.image.load(argument_handler().i).convert_alpha()   
    #scale image to fit screen if different size
    #window_image = pygame.transform.scale(window_image,(screen.get_width() + 500, screen.get_height()+500))
    
    #get image dimensions and center it in the middle of the screen
    image_rect = window_image.get_rect(center=(screen.get_width() // 2,screen.get_height() // 2))   
     
    #cv.x = width height image.shape = height, width(cv2 uses numpy for this)
    #img = cv2.imread(argument_handler().i, flags = cv2.CV_8UC4) 
    #depth_map = cv2.imread('depth.png')
    #depth_map = cv2.cvtColor(depth_map,cv2.COLOR_RGB2GRAY)
    #img = cv2.resize(img, np.flip(depth_map.shape[:2])) #height, width need to flip, ignore channel 3  
    #layers = get_depth_layers(depth_map) # maybe it crashes beause it was in a while loop 
    main()



