# Exp 7 Stock Market Prediction using Linear Regression and Real-Time Sentiment Analysis of Tweets

Date:07/09/2026

## AIM:

To implement **Stock Market Prediction using Linear Regression** to predict future stock prices using machine learning regression techniques and **Real-Time Sentiment Analysis of Tweets** to analyze user-provided text data such as tweets or reviews and classify their sentiment.

## DESIGN STEPS:

### Step 1:

Clone the repository from GitHub.

### Step 2:

Create a Python project in the preferred IDE (VS Code/PyCharm/Jupyter Notebook).

### Step 3:

Create the Python program for implementing Stock Market Prediction using Linear Regression and Real-Time Sentiment Analysis using suitable Python libraries.

### Step 4:

Load the historical stock market dataset and select the required features such as **Open, High, Low, Volume**, and **Close** price.

### Step 5:

Preprocess the stock market dataset and split it into training and testing data.

### Step 6:

Train the **Linear Regression** model using the training data and predict the stock prices for the testing data.

### Step 7:

Evaluate the stock prediction model using suitable regression metrics and visualize the actual and predicted stock prices.

### Step 8:

Accept user-provided text data such as **tweets, reviews, or comments** as input for sentiment analysis.

### Step 9:

Preprocess the input text by converting it into lowercase and removing unnecessary characters and punctuation.

### Step 10:

Calculate the **sentiment polarity score** of the input text using Natural Language Processing techniques.

### Step 11:

Classify the input text as **Positive, Negative, or Neutral** based on the polarity score.

### Step 12:

Execute the program and analyze the stock price prediction and sentiment analysis results.

## PROGRAM:
a) Stock Market Prediction using Linear Regression : To predict future stock prices using machine learning regression techniques.
```
from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("stock_data_big.csv")

print(data.head())

X = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nMean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

plt.figure(figsize=(8,5))
plt.plot(y_test.values[:50], label='Actual Prices', color='blue')
plt.plot(predictions[:50], label='Predicted Prices', color='red')
plt.title("Stock Price Prediction (Actual vs Predicted)")
plt.xlabel("Time")
plt.ylabel("Stock Close Price")
plt.legend()
plt.show()
```
b) Real-Time Sentiment Analysis of Tweets: Perform real-time sentiment analysis on user-provided text data (tweets or reviews).
```
from google.colab import files
uploaded = files.upload()
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_csv("tweets_big.csv")

def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


data["Sentiment"] = data["text"].apply(get_sentiment)

sentiment_counts = data["Sentiment"].value_counts()
print(sentiment_counts)

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green','red','gray'])
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Tweets/Reviews")
plt.show()

print("\nSample Results:")
print(data[["text", "Sentiment"]].head())
```
## OUTPUT:
a) Stock Market Prediction using Linear Regression : To predict future stock prices using machine learning regression techniques.
```
         Date    Open    High     Low   Close   Volume
0  2020-01-01  249.82  256.45  247.87  255.06  3296247
1  2020-01-02  480.29  489.26  473.38  486.16  1939050
2  2020-01-03  392.80  399.34  384.40  396.70  2770492
3  2020-01-06  339.46  342.56  335.04  342.05  2555257
4  2020-01-07  162.41  163.63  154.43  159.44  3077299

Mean Squared Error: 10.208543210007566
R2 Score: 0.9992601033703946
```

<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/4371e1ad-e901-43d8-b706-4d80ea6995c6" />


b) Real-Time Sentiment Analysis of Tweets: Perform real-time sentiment analysis on user-provided text data (tweets or reviews).
```
Sentiment
Negative    39944
Positive    33336
Neutral     26720
Name: count, dtype: int64
```

<img width="558" height="442" alt="image" src="https://github.com/user-attachments/assets/9ed9803d-f4a6-46c0-b9bb-0a6d563c468f" />

```
Sample Results:
                                                text Sentiment
0                                   Watching TV now.   Neutral
1                      Just finished my lunch break.   Neutral
2  The new update is awesome, super smooth and fast.  Positive
3                      Just finished my lunch break.   Neutral
4      The service was slow and the staff were rude.  Negative
```


## RESULT:

The **Stock Market Prediction using Linear Regression** and **Real-Time Sentiment Analysis of Tweets** were implemented successfully. The Linear Regression model was used to predict stock prices using historical stock market data, while the sentiment analysis system successfully analyzed user-provided tweets or reviews and classified them as **Positive, Negative, or Neutral**.
