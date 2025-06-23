# Predictive-Regression-Modeling-for-Retail-Sales-Forecasting
# Rossmann 7-Day Sales Forecast Dashboard

An interactive Streamlit application for forecasting daily sales at Rossmann stores. Built with XGBoost, this dashboard enables business users to generate actionable sales predictions for the next seven days, visualize trends, and plan inventory.

---

## 🚀 Features

- **7-Day Forecast**  
  Predicts daily sales for any Rossmann store over a one-week horizon.
- **Holiday Adjustment**  
  Allows users to specify school‐ and state‐holiday dates to capture demand shifts.
- **Promotion Flagging**  
  Toggle today’s promotion status to see its impact on sales.
- **Interactive Visualization**  
  Clean, transparent line chart with custom y-axis ticks (in \$1,000 units), date formatting (DD-MM-YYYY), and gridlines.
- **Rich Insights Panel**  
  - Highest, lowest, and average forecasted sales  
  - Overall trend indicator (📈/📉/🔁)  
  - Tabular view of each day’s predicted sales  
- **Scalable Pipeline**  
  - Reuses a MinMaxScaler for numeric feature normalization  
  - Loads pre-trained XGBoost model (`model.pkl`) for instant predictions  
  - Clean feature engineering consistent with training pipeline  

---

## 📁 Repository Structure
/predictive-regression-modeling-for-retail-sales-forecasting
- ├── app.py # Streamlit dashboard application
- ├── model.pkl # Trained XGBoost model artifact
- ├── scaler.pkl # Fitted MinMaxScaler for numeric features
- ├── store.csv # Store metadata (type, assortment, competition, promotions)
- ├── requirements.txt # Python dependencies
- └── README.md # This documentation

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/rossmann-sales-forecast.git
   cd rossmann-sales-forecast
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify artifacts**  
   Ensure `model.pkl`, `scaler.pkl`, and `store.csv` are present in the project root.

---

## ▶️ Usage

Run the Streamlit app locally:
```bash
streamlit run app.py
```

- **Sidebar inputs**  
  1. **Store ID** (1–1115)  
  2. **Promo today** (Yes/No)  
  3. **Forecast start date**  
  4. **Select school and state holiday dates** in the 7-day window  

- **Main panel**  
  - **Line chart** of forecasted sales  
  - **Insight summary** (highest, lowest, average, trend)  
  - **Per-day forecast table**  

---

## 🔧 Feature Engineering

- **Numeric features** (MinMax-scaled):  
  `Store`, `DayOfWeek`, `Day`, `Month`,  
  `CompetitionDistance`, `CompetitionMonths`,  
  `Promo`, `Promo2`, `Promo2Months`, `PromoInMonth`

- **One-hot features**:  
  `StoreType_{a,b,c,d}`, `Assortment_{a,b,c}`,  
  `StateHoliday_{0,a,b,c}`, `SchoolHoliday_{0,1}`

---

## 📈 Model Training (for reference)

This model was trained on an aggregated dataset (`train.csv` + `store.csv`) with the following steps:

1. **Load & merge data**  
2. **Feature engineering** (date decomposition, competition/promo durations, one-hots)  
3. **MinMax scaling** of numeric features  
4. **XGBoost regression** with hyperparameter tuning  
5. **Evaluation** using RMSE and R²  
6. **Persist** `model.pkl` and `scaler.pkl` via `joblib`

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch  
   ```bash
   git checkout -b feature/YourFeature
   ```  
3. Commit your changes  
   ```bash
   git commit -m "Add feature description"
   ```  
4. Push to your fork  
   ```bash
   git push origin feature/YourFeature
   ```  
5. Open a Pull Request

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---
