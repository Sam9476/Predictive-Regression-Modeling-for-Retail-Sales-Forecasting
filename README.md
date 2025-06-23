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
