#!/usr/bin/env python 
from VideoStream import VideoStream 
from Tracker import Tracker
from KalmanFilter import KalmanFilter
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


def extension_check(param):
    base,ext = os.path.splitext(param)
    if ext.lower() not in ('.jpg', '.png'):
        raise argparse.ArgumentTypeError('file must be a .png or .jpg')
    return param

def argument_handler():
    parser = argparse.ArgumentParser(prog="Virtual window", description="User inputs an image and the program tracks the movement of his face to adjust the image creating a \"Window effect\" ")
    parser.add_argument("-i" , type= extension_check, help="image input", required=True)
    args = parser.parse_args()
    return args

def pllax_eff(window_image, depth_image, head_x, head_y):
    screen_width, screen_height = screen.get_size()
    window_width, window_height = image_rect.width , image_rect.height
    #Initialize empty surface for the effect
    pllax_eff_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)

    pllax_strength = 0.1 

    # Loop over each pixel to apply the depth offset
    for x in range(window_width):
        for y in range(window_height):
            # Get depth value at (x, y)
            depth_value = depth_image[x, y, 0]  # Use R-channel or average RGB if grayscale

            # Calculate offset based on head movement and depth
            offset_x = (head_x - screen_width // 2) * depth_value * pllax_strength
            offset_y = (head_y - screen_height // 2) * depth_value * pllax_strength

            # Boundaries for safe drawing
            target_x = min(max(int(x + offset_x), 0), window_width - 1)
            target_y = min(max(int(y + offset_y), 0), window_height - 1)

            # Get the color at this pixel and place it on the depth effect surface
            color = window_image.get_at((x, y))
            pllax_eff_surface.set_at((target_x, target_y), color)

    return pllax_eff_surface

base_z = 40

first = True
d1 = base_z
d2 = d1
d3 = d1
kf = KalmanFilter()
def main():  
    try:
        stream = VideoStream().start()
        tracker = Tracker(stream.frame).start() 
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
                d1 = head_z
                d2 = head_z
                d3 = head_z
            
            d3 = d2
            d2 = d1
            d1 = head_z
            d1 = (d1 + d2 + d3) / 3 
            head_z = d1

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
                        print('ESCAPE')
                        running = False
                        sys.exit()
                    if event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL: #in pygame window
                        stream.stop()
                        tracker.stop()
                        pygame.quit() 
                        print('INTERRUPT')
                        running = False
                        sys.exit()
            
            #head movement
            #replaced screen.get_x() with image rect.Problem when image res will be lower than screen 
            #when looking left we want the image to go right
            offset_x = (head_x - image_rect.width // 2) * 0.7 
            offset_y = (head_y - image_rect.height // 2) * 0.7
            
            
            #set minimum as 1 so it doesnt zoom out more than the image
            z_scale_factor = max(1 ,np.round(head_z/30,5))  
            
            #scaled_width = max(int(image_rect.width * z_scale_factor), image_rect.width)  
            #scaled_height = max(int(image_rect.height * z_scale_factor), image_rect.height)
            scaled_width = int(image_rect.width * z_scale_factor)   
            scaled_height = int(image_rect.height * z_scale_factor)
            
            scaled_image = pygame.transform.scale(window_image, (scaled_width,scaled_height))
        
            #replaced screen dims but reduces fov
            image_x = ((image_rect.width - scaled_width) // 2)  + int(offset_x)
            image_y = ((image_rect.height  - scaled_height) // 2) + int(offset_y)
            
            scaled_rect = scaled_image.get_rect(center=(1920 // 2, 1280 // 2)) 
            
            offset_x = (head_x - scaled_rect.width // 2) * 0.5  
            offset_y = (head_y - scaled_rect.height // 2) * 0.5
            
            scaled_rect = scaled_rect.move((int(offset_x), int(offset_y)))
            #pllax_eff_surface = pllax_eff(window_image, depth_image, head_x, head_y)
            
            screen.fill((0,0,0)) 
            screen.blit(scaled_image, (int(image_x),int(image_y)))   
            pygame.display.flip()
     
    except KeyboardInterrupt: #in console
        pygame.display.quit()
        sys.exit("aborted\n")

 
if __name__ == "__main__":    
    #pygame top left corner is 0,0
    pygame.init()
    flags= pygame.FULLSCREEN | pygame.RESIZABLE
    pygame.display.set_caption("Virtual Window")
    screen = pygame.display.set_mode((0,0), flags) #fit native resolution 
    window_image = pygame.image.load(argument_handler().i).convert_alpha()   
    #get image dimensions and center it in the middle of the screen
    image_rect = window_image.get_rect(center=(screen.get_width() // 2,screen.get_height() // 2))   
    

    
    depth_image = pygame.image.load('depth.png').convert_alpha()
    #normalize
    depth_image = pygame.surfarray.array3d(depth_image).astype(np.float32) / 255.0     
    main()



