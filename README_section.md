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

## 📞 Contact

For questions or feedback, please open an issue or reach out to **Your Name** at **your.email@example.com**.
