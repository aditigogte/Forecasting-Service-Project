## 👤 Author
**Aditi Gogte**  *Data Science (End-to-End Time Series Forecasting System with API)* [https://www.linkedin.com/in/aditi-gogte-375a01288/]

# 📈 DemandAI: End-to-End Forecasting Service

A production-ready backend service designed to forecast sales across multiple states. This project implements a **Model Tournament** strategy to ensure the highest accuracy for every unique location.

## 🚀 Key Features
* **Automated Tournament:** Compares SARIMA, Prophet, XGBoost, and LSTM (Deep Learning) for every state.
* **Feature Engineering:** Implements mandatory lags (1, 7, 30), rolling averages, and seasonality features.
* **REST API:** Built with FastAPI to serve real-time 8-week forecasts.
* **Glassmorphism UI:** A modern dashboard to visualize model performance and predictions.

## 📁 Project Structure
- `app/`: Contains the API logic and web assets.
- `data/`: Storehouse for the input Excel files.
- `models/`: Stores the trained winning models (`.pkl`).
- `main.py`: The entry point to train and run the tournament.

## 🛠️ Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

   ---
