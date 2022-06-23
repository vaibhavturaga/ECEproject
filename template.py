import csv
import matplotlib.pyplot as plt

with open('GOOG.csv', mode='r') as file:
    reader = csv.DictReader(file)

    open_price =[]
    date = []
    high_price = []
    low_price = []
    close_price = []
    adj_close_price = []
    volume = []

    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)

    for row in reader:
        print(row)
        open_price.append(row['Open'])
        date.append(row['Date'])
        high_price.append(row['High'])
        low_price.append(row['Low'])
        close_price.append(row['Close'])
        adj_close_price.append(row['Adj Close'])
        volume.append(row['Volume'])

        plt.cla()
        plt.plot(date, open_price, label = 'Opening Price')
        plt.plot(date, high_price, label = 'High Price')
        plt.plot(date, close_price, label = 'Close Price')
        plt.plot(date, adj_close_price, label = 'Adjusted Closing Price')

        plt.legend()
        plt.xticks(rotation=90)
        plt.ylabel('Stock Price')
        plt.title('Stock Price of Google')
        plt.xlabel('Date')
        plt.pause(0.0001)
    
 
    plt.show()
