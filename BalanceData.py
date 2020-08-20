import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy', allow_pickle=True)

lefts = []
rights = []
forwards = []
backwards = []

shuffle(train_data)

for data in train_data:
    lines = data[0]
    choice = data[1]

    if choice == [1,0,0,0]:
        lefts.append([lines,choice])
    elif choice == [0,1,0,0]:
        forwards.append([lines,choice])
    elif choice == [0,0,1,0]:
        rights.append([lines,choice])
    elif choice == [0,0,0,1]:
        backwards.append([lines,choice])

forwards = forwards[:len(lefts)][:len(rights)][:len(backwards)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
backwards = backwards[:len(forwards)]

final_data = forwards + lefts + rights + backwards
shuffle(final_data)

np.save('training_data_balanced.npy', final_data)
