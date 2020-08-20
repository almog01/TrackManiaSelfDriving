import numpy as np
from GrabScreen import grab_screen
import win32gui
import cv2
from DrawLanes import draw_lanes
from DirectInput import press_key, release_key, W, A, S, D
from GetKeys import key_check
import time
from PIL import Image
import pytesseract
import os
import tensorflow as tf
import random

# ----------------------------------------------------------------------------------------------- #   
def roi(image, vertices):
    # create a black window with image dimensions
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

# ----------------------------------------------------------------------------------------------- #
def process_img(image):
    original_image = image
    # convert image to grayscale
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # apply edge detectoion algorithm to processed image
    processed_img = cv2.Canny(processed_img, 200, 300)
    
    # smoothen the processed image
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    
    # vertices for roi function
    vertices = np.array([[0, 550], [0, 400], [420, 330], [730, 330],  [1024, 400], [1024, 550]])

    # apply mask to processed image
    processed_img = roi(processed_img, vertices)
    
    # detect lines in processed image
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 200, np.array([]),
                            minLineLength=200, maxLineGap=20)
    
    # draw lines on processed image                               
    l1 = (0, 0, 0, 0)
    l2 = (0, 0, 0, 0)
    try:
        l1, l2, m1, m2 = draw_lanes(original_image,lines)
        if l1[0] < -15000 or l1[0] > 15000 or \
            l1[1] < -15000 or l1[1] > 15000 or \
            l1[2] < -15000 or l1[2] > 15000 or \
            l1[3] < -15000 or l1[3] > 15000 or \
            l2[0] < -15000 or l2[0] > 15000 or \
            l2[1] < -15000 or l2[1] > 15000 or \
            l2[2] < -15000 or l2[2] > 15000 or \
            l2[3] < -15000 or l2[3] > 15000:
            raise Exception()
        
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 10)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 10)

    except Exception as e:
        l1 = (0, 0, 0, 0)
        l2 = (0, 0, 0, 0)
        return gray_img, original_image, l1, l2
    
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 2)
            except Exception as e:
                pass
    except Exception as e:
        pass

    return gray_img, original_image, l1, l2

# ----------------------------------------------------------------------------------------------- #
def get_car_speed(gray):
    gray = Image.fromarray(gray)
    cropped = gray.crop((875, 590, 965, 660))  # (left, top, right, bottom) - trackmania
    gray = np.array(cropped)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.resize(gray, (int(gray.shape[0]/2), int(gray.shape[1]/3.5)))
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
    gray = cv2.bitwise_not(gray)

    # load the image as a PIL/Pillow image, apply OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Avsha\AppData\Local\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(Image.fromarray(gray), lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=0123456789")

    return text

# ----------------------------------------------------------------------------------------------- #
def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D,S] boolean values.
    '''
    output = [0,0,0,0]

    if 'S' in keys:
        output = [0,0,0,1]
    elif 'A' in keys:
        output = [1,0,0,0]
    elif 'D' in keys:
        output = [0,0,1,0]
    elif 'W' in keys:
        output = [0,1,0,0]

    return output

# ----------------------------------------------------------------------------------------------- #
file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    training_data = []

# ----------------------------------------------------------------------------------------------- #
def main():

    paused = False
    while True:
        
        keys = key_check()
        if not paused:
            # process window
            bgr_screen = grab_screen("ManiaPlanet")
            
            # convert input screen from BRG to RGB
            bgr_screen = cv2.cvtColor(bgr_screen, cv2.COLOR_BGR2RGB)

            gray_img, original_image, l1, l2 = process_img(bgr_screen)

            if l1 != None and l2 != None and l1 != (0, 0, 0, 0) and l2 != (0, 0, 0, 0):

                speed = get_car_speed(gray_img)
                if speed == '':
                    continue
                speed = int(speed)

                key_pressed = keys_to_output(keys)
                if key_pressed != [0,0,0,0]:
                    inputs = [speed, l1[0], l1[1], l1[2], l1[3], l2[0], l2[1], l2[2], l2[3]]
                    training_data.append([inputs, key_pressed])

                    if len(training_data) % 1000 == 0:
                        print(len(training_data))
                        np.save(file_name, training_data)

            # show window
            # cv2.imshow('gray', gray_img)
            # cv2.imshow('game', bgr_screen)
            # cv2.imshow('game2', original_image)

        
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Paused!')
                paused = True
                time.sleep(1)
        if 'Q' in keys:
            print('Finished!')
            return
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
