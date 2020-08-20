import win32api as wapi
from DirectInput import press_key, release_key, W, A, S, D
import time
from ahk import AHK

keyList = []
for char in "SADWQT":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys