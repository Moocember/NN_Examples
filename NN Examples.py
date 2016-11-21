import numpy as np
np.random.seed(7)
import matplotlib.pyplot as plt
import pandas

"""
This project shows two distinct ways to perform time series prediction using neural networks.
It contains two main functions: timeSeriesNNFromScratch and timeSeriesPredictionKerasNN.

"timeSeriesNNFromScratch" is a function that I wrote from scratch using only numpy.
Its based on the classic forward and back propagation using gradient descent method to search for the global minima
.
"timeSeriesPredictionKerasNN" is a function that uses Keras, a very powerful machine learning library built upon Theano.
Futhur documentation on Keras can be found on their offical website: https://keras.io/
"""

#The variable kerasIsEnabled allows the code to run without Keras when set to False
try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM
    kerasIsEnabled = True
except:
    kerasIsEnabled = False


"""
Dropout regularization:
    Dropout is used to prevent overfitting by randomly removing a fixed percentage of neuron connections
Parameters
----------
    dropoutPercent: (float) Percentage of neuron connections removed at a time
    activation: (ndarray) The layer from which you would like to remove neuron connections
    hiddensize: (int) The size of the layer to which the randomly removed neuron connections are going towards
    X: (ndarray) The data you are training with
"""
def dropout(dropoutPercent: float,activation,hiddensize: int,X):
        activation*= np.random.binomial([np.ones((len(X),hiddensize))],1-dropoutPercent)[0] * (1.0/(1-dropoutPercent))
        return activation
"""
Sigmoid Activation:
    Sigmoid is the nonlinear activation used in this network. It is used to connect layers to each other.
    Using a nonlinear activation allows the NN to find more subtle relationships between neurons than a linear activation
Parameters
----------
    x: (ndarray) The data you would like to activate
    deriv = False: (bool) The sigmoid function to be used in forward propagation
    deriv = True: (bool) The derivative of the sigmoid function to be used in back propagation
"""
def sigmoid(x, deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
"""
Time Series Neural Network from Scratch:
    This function uses an artificial neural network to predict any number of periods ahead
Parameters
----------
    data: (ndarray) The data parameter is used to retrieve your dataset
    predictionPeriod: (int) The number of periods ahead you want to predict
    alpha: (float) The learning rate of each epoch
    secondLayerDim: (int) The number of neurons in the second layer (first hidden layer)
    thirdLayerDim: (int) The number of neurons in the third layer (second hidden layer)
    numEpochs: (int) The number of times you want to adjust your thetas
    graphError: (bool) Graph the mean squared error of each epoch when set to True (Significatly increases run time)
"""
def timeSeriesNNFromScratch(data,predictionPeriod: int,alpha: float,secondLayerDim: int,thirdLayerDim: int,numEpochs: int, graphError = True):
    #Normalizing the dataset to the range (0,1) to match the range of the sigmoid activation
    maxY = max(data)
    minY = min(data)
    dataset = (data-minY)/(maxY-minY)

    #Offsetting X and y to predict any number of periods into the future
    X = dataset[:-predictionPeriod]
    y = dataset[predictionPeriod:]

    #Adding a bias
    X = np.c_[np.ones((len(X),1)),X]

    #Split data into training and cross validation sets
    train_size = int(len(y) * 0.67)
    trainX, validationX = X[0:train_size, :], X[train_size - 1:, :]
    trainY, validationY = y[0:train_size, :], y[train_size - 1:, :]
    trainY=trainY.reshape(len(trainY),1)
    validationY = validationY.reshape(len(validationY), 1)

    # Randomly initialize the thetas with mean 0
    syn0 = 2 * np.random.random((len(X.T), secondLayerDim)) - 1
    syn1 = 2 * np.random.random((secondLayerDim, thirdLayerDim)) - 1
    syn2 = 2 * np.random.random((thirdLayerDim, 1)) - 1

    #Used for graphing validation and training error
    if graphError:
        trainingErrorHistory = list()
        validationErrorHistory = list()
    epoch = np.copy(numEpochs)
    #Training
    for i in range(numEpochs):
        epoch-=1
        print("Epoch: ",epoch,"/", numEpochs)

        #Forward Propagation
        a0 = trainX
        a1 = sigmoid(np.dot(a0, syn0))
        a1 = dropout(.2,a1,secondLayerDim,trainX)
        a2 = sigmoid(np.dot(a1, syn1))
        a3 = sigmoid(np.dot(a2, syn2))

        #Back Propagation
        a3err = a3 - trainY
        a3Delta = np.multiply(a3err, sigmoid(a3, deriv=True))

        a2err = np.dot(a3Delta, syn2.T)
        a2Delta = np.multiply(a2err, sigmoid(a2, deriv=True))

        a1err = np.dot(a2Delta, syn1.T)
        a1Delta = np.multiply(a1err, sigmoid(a1, deriv=True))

        #Adjusting the thetas
        syn2 -= alpha * (a2.T.dot(a3Delta))
        syn1 -= alpha * (a1.T.dot(a2Delta))
        syn0 -= alpha * (a0.T.dot(a1Delta))

        if graphError:
            a0Validation = validationX
            a1Validation = sigmoid(np.dot(a0Validation,syn0))
            a2Validation = sigmoid(np.dot(a1Validation,syn1))
            a3Validation = sigmoid(np.dot(a2Validation,syn2))
            a3errValidation = a3Validation - validationY
            trainingErrorHistory.append(sum((a3err**2)/len(a3err)))
            validationErrorHistory.append(sum((a3errValidation**2)/len(a3errValidation)))

    if graphError:
        trainingErrorHistory = np.array(trainingErrorHistory).reshape((len(trainingErrorHistory),1))
        validationErrorHistory = np.array(validationErrorHistory).reshape((len(validationErrorHistory),1))
        plt.plot(trainingErrorHistory)
        plt.plot(validationErrorHistory)
        plt.title('ANN From Scratch Error History')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epoch')
        plt.legend(['Training Error', 'Validation Error'], loc='upper left')
        plt.show()


    #Using the model to predict the cross validation data
    a0 = validationX
    a1 = sigmoid(np.dot(a0,syn0))
    a2 = sigmoid(np.dot(a1,syn1))
    ho = sigmoid(np.dot(a2,syn2))

    #Transforming the data back to its original range
    ho = ho*(maxY-minY)+minY
    validationY = validationY*(maxY-minY)+minY

    #Plotting the hypothesis and Cross Validation y
    plt.plot(validationY[:-predictionPeriod])
    plt.plot(ho[predictionPeriod:]+np.mean(validationY)-np.mean(ho))
    plt.title('ANN from Scratch Cross Validation Plot')
    plt.legend(['Hypothesis', 'Actual'], loc='upper left')
    plt.show()
    return ho

"""
Time Series Prediction Using the Keras Library:
    This function uses the Keras library to create a recurrent neural network that predicts any number of periods ahead
Parameters
----------
    data: (ndarray) The data parameter is used to retrieve your dataset
    predictionPeriod: (int) The number of periods ahead you want to predict
    recurrentLayerDim: (int) The number of neurons in your recurrent layer
    numEpochs: (int) The number of times you want to adjust your thetas
"""
def timeSeriesPredictionKerasNN(data,predictionPeriod: int,recurrentLayerDim: int,numEpochs: int):
    #Normalizing the dataset to the range (0,1) to match the range of the "relu" activation
    maxY = max(data)
    minY = min(data)
    dataset = (data-minY)/(maxY-minY)

    #Offsetting X and y to predict "predictionPeriod" periods into the future
    X = dataset[:-predictionPeriod]
    y = dataset[predictionPeriod:]

    #Split data into training and cross validation sets
    train_size = int(len(y) * 0.67)
    look_back = 1
    trainX, validationX = X[0:train_size, :], X[train_size - look_back:, :]
    trainY, validationY = y[0:train_size, :], y[train_size - look_back:, :]
    trainY=trainY.reshape(len(trainY),1)
    validationY = validationY.reshape(len(validationY), 1)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validationX = np.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(recurrentLayerDim,input_dim=len(X.T), activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit(trainX, trainY, validation_data=(validationX, validationY), nb_epoch=numEpochs, batch_size=1, verbose=2)

    #Using the model to predict the cross validation data
    ho = model.predict(validationX)

    #Transforming the data back to its original range
    ho = ho*(maxY-minY)+minY
    validationY = validationY*(maxY-minY)+minY

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Keras RNN Error History')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['Training Error', 'Validation Error'], loc='upper left')
    plt.show()

    #Plotting the hypothesis and Cross Validation y
    plt.plot(validationY[:-predictionPeriod])
    plt.plot(ho[predictionPeriod:]+np.mean(validationY)-np.mean(ho))
    plt.title('Keras Prediction Cross Validation Plot')
    plt.legend(['Hypothesis', 'Actual'], loc='upper left')
    plt.show()
    return ho


#Using international airline passengers data as an example because my current boss doesn't want me sharing features
if __name__ == '__main__':
    dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    if kerasIsEnabled:
        kerasHo = timeSeriesPredictionKerasNN(dataset,1,4,150)
    scratchHo = timeSeriesNNFromScratch(dataset,1,.01,10,10,20000)

