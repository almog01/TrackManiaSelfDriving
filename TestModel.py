import numpy as np
from GrabScreen import grab_screen
import cv2
import time
from DirectInput import press_key, release_key, W, A, S, D
from ArtificialNeuralNetwork import artificial_neural_network
from GetKeys import key_check
import pytesseract
from PIL import Image
from DrawLanes import draw_lanes

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
release_time = 0.1
def straight():
    press_key(W)
    time.sleep(0.3)
    release_key(W)
    release_key(A)
    release_key(D)
    release_key(S)

def left():
    press_key(A)
    press_key(W)
    time.sleep(release_time)
    release_key(W)
    release_key(A)
    release_key(D)
    release_key(S)

def right():
    press_key(D)
    press_key(W)
    time.sleep(release_time)
    release_key(W)
    release_key(D)
    release_key(A)
    release_key(S)

def brake():
    press_key(S)
    time.sleep(release_time)
    release_key(W)
    release_key(D)
    release_key(A)
    release_key(S)

# ----------------------------------------------------------------------------------------------- #
def main():
    LR = 1e-3
    num_of_data = np.load('training_data_balanced.npy', allow_pickle=True)
    model_name = 'ManiaPlanet-v1.0-{}.model'.format(num_of_data.shape[0])
    model = artificial_neural_network(1, 9, LR, 4)
    model.load(model_name)

    # for i in list(range(2))[::-1]:
    #     print(i+1)
    #     time.sleep(1)
        
    paused = False
    while(True):
        if not paused:
            bgr_screen = grab_screen("ManiaPlanet")
            bgr_screen = cv2.cvtColor(bgr_screen, cv2.COLOR_BGR2RGB)
            gray_img, original_image, l1, l2 = process_img(bgr_screen)

            speed = get_car_speed(gray_img)
            if speed == '':
                speed = '50'
            speed = int(speed)
            
            inputs = [speed, l1[0], l1[1], l1[2], l1[3], l2[0], l2[1], l2[2], l2[3]]
            inputs = np.array(inputs).reshape(1, 9, 1)
            pred = model.predict([inputs])[0]
            max_val = np.argmax(pred)
            if l1 == (0, 0, 0, 0) or l2 == (0, 0, 0, 0):
                max_val = 1

            if max_val == 0:
                left()
                print("Left")
            elif max_val == 1:
                straight()
                print("Straight")
            elif max_val == 2:
                right()
                print("Right")
            elif max_val == 3:
                brake()
                print("Brake")
        
        keys = key_check()

        if 'T' in keys:
            if paused:
                print("Unpaused!")
                paused = False
                time.sleep(1)
            else:
                print("Paused!")
                paused = True
                release_key(A)
                release_key(W)
                release_key(D)
                release_key(S)
                time.sleep(1)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Finished!")
            cv2.destroyAllWindows()
            break

main()