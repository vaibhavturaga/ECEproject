import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Importing dataset for apple stock
df_apple = pd.read_csv("AAPL.csv")

# create training and testing set
data_set = df_apple[["Close"]]
train_length = (int) (len(data_set) * .8)
trainset = data_set[:train_length]
testset = data_set[train_length:]

#scale data to maximize models accuracy
scaler = MinMaxScaler(feature_range = (0,1))
trainset = scaler.fit_transform(trainset)
testset = scaler.fit_transform(testset)

#format x and y sets to be in proper format for LSTM
xtrain = []
ytrain = []

#use time step of 10 days to get training data (Model will use previous 10 days to predict 11th day, and so on)
for i in range(len(trainset) - 11):
    xtrain.append(trainset[i:i+10,0])
    ytrain.append(trainset[i+11,0])

#set as numpy array(allows us to reshape data easily)
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

#reshape training data (defualt np array has size (size,) rather than (size, 1) which is required for model) 
xtrain = np.reshape(xtrain, (len(ytrain), 10))
print(xtrain.shape)
#set model type as Sequential (data is a sequence of numbers)
model = Sequential()
#calculate optimal number of units to be used in lstm creation (Units can be pretty much anything, though this formula generally works well)
unit = (int) (len(xtrain) / (10))
#add LSTM to model using keras -> activation function is rectified linear unit
model.add(LSTM(unit, activation = 'relu', input_shape = (10,1)))
#set number of outputs from neural network to be 1
model.add(Dense(1))

#compile model using 'adam', which is generally known to work the best for data like this
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

#get summary of model created
model.summary()
#fit model to training data using 2 epochs (tested this with multiple epochs and loss began to stabalize after the first epoch)
model.fit(xtrain, ytrain, epochs = 2, batch_size = 1)
#save model so that future tests don't require recompiling of model
model.save("model.h5")

#create testing data
xtest = np.empty(0)
print(testset[0,0])
for i in range(len(testset) - 11):
    xtest = np.append(xtest, testset[i:i+10,0])
    print(testset[i:i+10,0])


#run test data through model
dim = (int) (xtest.shape[0] / 10)
xtest = np.reshape(xtest, (dim, 10))
prediction = model.predict(xtest, batch_size = 1)
final_prediction = scaler.inverse_transform(prediction)

#Align predictions with date
date = df_apple["Date"]
final_prediction = np.reshape(final_prediction, (len(final_prediction),))
testset = scaler.inverse_transform(testset)
testset = np.reshape(testset, (len(testset),))

#calculate mse (used to determine which stock the model is better at predicting)
mse = mean_squared_error(testset[-len(final_prediction):], final_prediction)
print("The mean squared error is:")
print(mse)
#plot predictions for next day prediction vs actual for entire test set
plt.plot(date[-len(final_prediction):], final_prediction, label = "Prediction")
plt.plot(date[-len(final_prediction):], testset[-len(final_prediction):], label = "Actual")
plt.xticks(rotation = 90)
plt.legend()
plt.show()
#plot next day prediction from previous 60 days
plt.plot(date[-60:], final_prediction[-60:], label = "Prediction")
plt.plot(date[-60:], testset[-60:], label = "Actual")
plt.xticks(rotation = 90)
plt.show()
