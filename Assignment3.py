import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model




def generate_data(num_of_data):
    X = np.random.randint(-20, 20, size = num_of_data, dtype = int)
    Y = 5*X**3 - 68*X**2 - 7*X + 1 
    return X, Y	


def DNN_Model():
	inputs = Input(shape=(1))
	x = Dense(32, activation='relu')(inputs)
	x = Dense(64, activation='relu')(x)
	x = Dense(128, activation='relu')(x)
	outputs = Dense(1)(x)
	
	model = Model(inputs = inputs, outputs = outputs)
	
	return model


def main():

	data_x, data_y = generate_data(5000)
	data_x = data_x / max(data_x)
	data_y = data_y / max(data_y)
	
	train_data_x = data_x[:int(len(data_x)*.9)]
	train_data_y = data_y[:int(len(data_y)*.9)]
	
	valid_x = data_x[int(len(data_x)*.9):int(len(data_x)*.95)]
	valid_y = data_y[int(len(data_y)*.9):int(len(data_y)*.95)]
	
	test_x = data_x[int(len(data_x)*.95):]
	test_y = data_y[int(len(data_y)*.95):]
	
	
	model = DNN_Model()
	model.summary()
	
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss="mean_squared_error", metrics=['accuracy'])
	history = model.fit(train_data_x, train_data_y, epochs=10, validation_data=(valid_x, valid_y))
	
	
	performance_dict = history.history
	plt.figure(figsize = (12, 12))
	plt.subplot(2,2,1)
	plt.plot(performance_dict['loss'])
	plt.plot(performance_dict['val_loss'])
	plt.legend(['train_loss', 'validation_loss'])
	
	plt.subplot(2,2,2)
	plt.plot(performance_dict['accuracy'])
	plt.plot(performance_dict['val_accuracy'])
	plt.legend(['train_accuracy', 'validation_accuracy'])
	
	plt.subplot(2,2,3)
	plt.scatter(test_x, test_y)
	
	plt.subplot(2,2,4)
	plt.scatter(test_x, test_y)
	plt.scatter(test_x, model.predict(test_x))
	
	plt.show()
	
	

if __name__ == '__main__':
	main()