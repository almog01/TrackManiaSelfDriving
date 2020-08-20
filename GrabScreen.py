import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(window_name):

    # Detect the position of The window 'racer'
    windows_list = []
    toplist = []
    def enum_win(hwnd, result):
        win_text = win32gui.GetWindowText(hwnd)
        windows_list.append((hwnd, win_text))
    win32gui.EnumWindows(enum_win, toplist)
    game_hwnd = 0
    for (hwnd, win_text) in windows_list:
        if window_name in win_text:
            game_hwnd = hwnd
            break
    region = win32gui.GetWindowRect(game_hwnd)

    left,top,x2,y2 = region
    window_header = 23
    margin = 3
    left += margin
    top += window_header + margin
    width = x2 - left - margin
    height = y2 - top - margin

    hwin = win32gui.GetDesktopWindow()
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)