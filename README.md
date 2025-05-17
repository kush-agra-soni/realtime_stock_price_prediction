# **Realtime Stock Price Prediction**

## **Business Context**

Accurate stock price prediction is crucial in financial markets as it influences investment decisions, risk management, and portfolio optimization. This project utilizes **Long Short-Term Memory (LSTM)** networks for predicting stock prices, demonstrating the application of deep learning techniques in financial forecasting.

---

## **Practical Application**

Stock price prediction serves multiple stakeholders, including:

- **Investors**: Make informed decisions and identify profitable opportunities.
- **Traders**: Adapt their strategies based on predicted price movements.
- **Financial Analysts**: Analyze market trends and provide actionable insights.

By accurately forecasting stock prices, these stakeholders can better manage risks and optimize their portfolios.

---

## **Challenges and Limitations**

Stock price prediction faces several challenges:

- **Market Volatility**: Prices can fluctuate unpredictably due to external events and market sentiment.
- **Data Noise**: Financial data is often noisy, with irregularities that complicate forecasting.
- **External Factors**: Global events, economic policies, and geopolitical factors can significantly impact stock prices.

This project acknowledges these challenges and demonstrates how **LSTM** networks can model the temporal dependencies within financial data to improve predictions.

---

## **Objective**

The project aims to achieve the following goals:

- Improve forecasting accuracy for stock prices.
- Demonstrate the effective use of **LSTM** networks with various hyperparameters.
- Provide insights into how deep learning models can capture temporal dependencies in time series data.

---

## **Data Description**

The dataset includes historical stock prices for **Apple Inc. (AAPL)**, retrieved using the [Yahoo Finance API](https://pypi.org/project/yfinance/). The key features of the dataset are:

- **Open**: The price of the stock at market open.
- **Close**: The price of the stock at market close.
- **High**: The highest price during the trading day.
- **Low**: The lowest price during the trading day.
- **Volume**: The number of shares traded.

---

## **Approach**

### 1. **Neural Network Basics**
   - Understand the fundamentals of LSTM networks for handling sequential data.

### 2. **Load Stock Data**
   - Retrieve historical stock price data using the **Yahoo Finance API**.

### 3. **Data Preprocessing**
   - Normalize and scale the stock price data.
   - Create sliding windows for time series analysis to model dependencies.

### 4. **LSTM Model Development**
   - Implement LSTM networks to capture long-term dependencies in the data.

### 5. **Incorporate Technical Indicators**
   - Enhance predictions by incorporating additional features such as technical analysis indicators.

### 6. **Model Evaluation**
   - Evaluate the performance of the LSTM models using appropriate metrics.
   - Generate future stock price sequences for forecasting.

---


## Modular Code Structure

```
.  
├── core/
│   ├── app.py                                   # Functions for frontend (Streamlit).
│   ├── train.py                                 # Functions for training LSTM model.
├── utils/                                       # Folder with modular Python scripts.
│   ├── install_req.py                           # Functions for automatic installtion for req txt and pip.
│   ├── engine.py                                # Main script to execute the models training (Optional Modules).
├── .gitignore                                   # To ignore the temp files like .venv and pycache folders.
├── LICENSE                                      # License for this repo
├── start.py                                     # Single click runable file for initial start.
└── README.md                                    # Project documentation.
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Create .venv and install Dependencies

Install every required thing using:

```bash
>>>python start.py
```

---
Deployed Version - [Link](https://predistocks.streamlit.app/)

---

## Results

- **Forecasting Accuracy**:
  - LSTM models effectively captured temporal patterns in stock prices.
- **Technical Indicator Integration**:
  - The integration of additional features improved prediction accuracy.
- **Deep Learning Insights**:
  - The model demonstrated the ability to predict future prices, with results visualized using candlestick charts similar to real-world stock market data.
  

---


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or suggestions, please reach out to:

- **Name**: Kushagra Soni
- **Email**: mailme.kushagrasoni6446@gmail.com
- **GitHub**: [Kushagra Soni](https://github.com/kush-agra-soni)

---

