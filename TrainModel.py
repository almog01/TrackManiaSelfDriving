import numpy as np
from ArtificialNeuralNetwork import artificial_neural_network

LR = 1e-3
EPOCHS = 10

model = artificial_neural_network(1, 9, LR, 4)

train_data = np.load('training_data_balanced.npy', allow_pickle=True)

model_name = 'ManiaPlanet-v1.0-{}.model'.format(train_data.shape[0])

partition = int(train_data.shape[0] * 0.2)
train = train_data[:-partition]
test = train_data[-partition:]

X = np.array([i[0] for i in train]).reshape(-1, 1, 9, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, 1, 9, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS,
    validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)

model.save(model_name)

# tensorboard --logdir=foo:C:\Users\Avsha\log