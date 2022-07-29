import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
'''
Here is a demo code to create a next day prediction dataset for APPL that you might find useful
'''

# Importing dataset for apple stock
df_apple = pd.read_csv("AAPL.csv")
# Removes features unnecessary to prediction of closing price
remove_features = ["Open", "High", "Low", "Adj Close", "Volume"]
# Create the "Prediction" column by shifting the "Close" column down by one row
df_apple["Prediction"] = pd.Series(np.append(df_apple["Close"][1:].to_numpy(), [0]))
df_apple.drop(df_apple.tail(1).index, inplace=True)
df_apple.drop(remove_features, axis=1, inplace=True)

df_apple["Date"] = pd.to_datetime(df_apple["Date"])
df_apple["Date"] = pd.to_numeric(df_apple["Date"])
# Create X,y from the dataset
x = np.array(df_apple.drop(["Close"], axis=1))

train_length = (int) (len(x) * .8)
x_train = x[:train_length]
x_test = x[train_length:]

print(x_train.shape)

#Scale data 
scale_data_zero_to_one = MinMaxScaler()
scale_data_zero_to_one.fit(x_train)
scaled_x_train = scale_data_zero_to_one.transform(x_train)
scaled_x_test = scale_data_zero_to_one.transform(x_test)
#create model
time_generator = TimeseriesGenerator(scaled_x_train[:,0], scaled_x_train[:,1], length = 20, batch_size = 1)
#used formula Number of Samples / (scaling factor * (Number of Input + Output Neurons))
unit = (int) (len(x_train) / (10))

model = Sequential()
model.add(LSTM(units = unit, activation = 'relu', input_shape = (2, )))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()
model.fit(time_generator, epochs = 2)

#predict
input_length_left = 20
predicted_values = []
data = scaled_x_test[-input_length_left:]
data_test = []
for i in data:
    data_test.append(i[1])
print(data_test)
print("------------------------")
print(data)
for i in range(len(x_test)):
    data_test = np.reshape(data_test, (20, 1))
    pred = model.predict(data_test)[0]
    predicted_values.append(pred)
    print(data_test)
    print("-------------------------")
    print(pred)
    print("--------------------------")
    data_test = np.delete(data_test, 0)
    data_test = np.append(data_test, pred)

print(predicted_values)
prediction = scale_data_zero_to_one.inverse_transform(predicted_values)
plt.plot(df_apple["Date"], prediction)
plt.plot(df_apple["Date"], df_apple["Close"])
print(prediction)