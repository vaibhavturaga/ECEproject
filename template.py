import csv
import matplotlib.pyplot as plt

with open('project-vaibhavturaga\AAPL.csv', mode='r') as file:
    reader = csv.DictReader(file)
    line = 0
    for row in reader:
        print(row)
        open_price = row['Open']
        date = row['Date']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        adj_close_price = row['C']
        plt.plot(date, float(open_price))
        plt.pause(0.0001)
    
    plt.show()
