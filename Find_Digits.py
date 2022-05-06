# Jason Matuszak 

import os
import cv2
import sys
import math
import time
import random
# import rgb1602
# import serial
import numpy as np
# sys.path.append('../')
import PIL.Image as pil
# import RPi.GPIO as GPIO
from array import *
# from time import sleep
# from picamera import PiCamera
from os.path import expanduser
from PIL import Image, ImageOps

def purge_directory(directory, choice):
    # Read the directory and count the number of images
    files = 0
    for path in os.listdir(directory):
        # check if current path is a file
        if os.path.isfile(os.path.join(directory, path)):
            files += 1
     # Purge output directory
    if (choice == 1):
        for y in range(files):
            if (y+1 <= files ):
                os.remove(f"{directory}/Hand_Written_Digit_" + str(y+1) + ".jpg")      
        print(str(files) + ' old images removed from mnist image directory')
    
    elif (choice == 2):
        for y in range(files):
            if (y+1 <= files ):
                os.remove(f"{directory}/image" + str(y+1) + ".jpg")      
        print(str(files) + ' old images removed from CNN image directory')
    return None

# Image padding function for use after cropping digits
def pad_image(image, expected_size):
    image.thumbnail((expected_size[0], expected_size[1]))
    
    width = expected_size[0] - image.size[0]
    height = expected_size[1] - image.size[1]
    
    pad_width = width // 2
    pad_height = height // 2
    
    padding = (pad_width, pad_height, width - pad_width, height - pad_height)
    
    return ImageOps.expand(image, padding)

def find_digits(image):
    # Image saving directories
    saving_path = '/home/pi/Desktop/Cropped_Digits'
    binary_image_path = '/home/pi/Desktop/Binary_Images'
    boxed_digits_path = '/home/pi/Desktop/Boxed_Digits'
    number_of_digits = 0
    
    # Clean up output directory
    purge_directory(saving_path,1)
    
    # Resizing image for better thresholding results
    image = cv2.resize(image,(500,500))
   
    # Create input image copy
    imageCopy = image.copy()

    # Grayscale the input image
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Obtain binary image through adaptive thresholding, values may vary depending on lighting
    ret, binaryImage = cv2.threshold(grayscaleImage,90, 255, cv2.THRESH_BINARY_INV) #65-90 best
    
    # Convert from numpy to pil
    image_pil_crop = Image.fromarray(binaryImage)
    
    #Crop image slightly to compensate for leg/camera positioning on enclosure
    im_pil_crop = image_pil_crop.crop((25, 25, 475, 475))
    
    # Convert from numpy to pil
    image_box = Image.fromarray(image)
    
    # Crop image for correct bounding box positioning when drawing
    image_box = image_box.crop((25,25,475,475))
    image_box = np.array(image_box)
    image_crop_num = np.array(im_pil_crop)
    binaryImage_crop = image_crop_num
    
    # Save binary image 
    cv2.imwrite(os.path.join(binary_image_path , 'Binary_Image.jpg'), binaryImage_crop)
    
    # Obtain binary image contours
    conts, hier = cv2.findContours(binaryImage_crop.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = [cv2.boundingRect(cont) for cont in conts]
    j = 0
    for j,digit in enumerate(digits):
        # Draw bounding boxes around each digit on the original image and save it
        cv2.rectangle(image_box, (digits[j][0] , digits[j][1]), (digits[j][0] + digits[j][2], digits[j][1] + digits[j][3]), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(boxed_digits_path ,'Boxed_Digits.jpg'), image_box)
        for i in range(len(digits)):
            # Get the roi for each bounding rectangle:
            x, y, w, h = digits[i]
            croppedImg = binaryImage_crop[y:y + h, x:x + w]
            # Convert from numpy array to PIL
            image_mnist_pil = Image.fromarray(croppedImg)
            # Pad image with 0s 
            image_mnist_pil = pad_image(image_mnist_pil,(100,300))
             # Resize to mnist format
            image_mnist_pil = image_mnist_pil.resize((28,28))
                        
            width, height = image_mnist_pil.size
            # Filter extra noise out of the cropped images
            for x in range(0,width):
                for y in range(0,height):
                    if image_mnist_pil.getpixel((x,y)) > 1:
                        image_mnist_pil.putpixel((x,y),255)
            # Save final cropped digits (Up to 10)
            if (i+1 < 11):
                image_mnist_pil.save(f"{saving_path}/Hand_Written_Digit_" + str(i+1) + ".jpg")
                number_of_digits = i
            
    return number_of_digits

if __name__ == '__main__':
    while (1):
        '''
        Arduino serial communication set up
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        ser.reset_input_buffer()
        take_picture_switch()
        '''
        start_time = time.time()
        image = cv2.imread('/home/pi/Desktop/Pi_Pics/image.jpg')
        # image = take_picture()
        digits = find_digits(image)
        cv_time = time.time() - start_time
        print("Computer Vision finished in: " + str(cv_time) + " Seconds")
        '''
        inferencing(digits)
        ml_time = time.time() - start_time
        print("Machine Learning finished in: " + str(time.time() - start_time) + " Seconds")
        lcd_digits()
        '''
        total_time = time.time() - start_time
        print("Total elapsed time: " + str(total_time) + " Seconds")
        '''
        # Write timing values
        open('Time.txt', 'w').close()
        with open('Time.txt', 'w') as f:
            f.write(str(cv_time))
            f.write('\n')
            f.write(str(ml_time))
            f.write('\n')
            f.write(str(total_time))
            f.write('\n')
        print("Done...waiting for another picture")
        '''
        start_time = 0.0