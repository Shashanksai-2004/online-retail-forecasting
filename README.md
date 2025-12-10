📊 Online Retail Revenue Analysis & Forecasting
A complete end-to-end data analytics and forecasting project built using Python. This project analyzes an online retail dataset to uncover customer behavior, product performance, country-level purchasing trends, and revenue seasonality. It also includes advanced forecasting models to predict future revenue.
________________________________________
🔍 Project Overview
This project performs: - Data cleaning & preprocessing of raw transactional retail data. - Exploratory Data Analysis (EDA) including trend analysis, heatmaps, seasonality breakdowns, and customer segmentation. - Statistical Analysis (T-tests, ANOVA, Chi-square, skewness/kurtosis, outlier detection). - Feature Engineering for forecasting (lags, rolling windows, date features). - Forecasting Models using Prophet, SARIMAX, and LightGBM. - Business Insights for inventory planning, marketing decisions, and customer targeting. - BI-ready datasets for Power BI/Tableau dashboards.
________________________________________
🗂 Project Structure
online-retail-forecasting/
│── Online_Retail.py              # Full project code
│── cleaned_online_retail.csv     # Cleaned dataset
│── daily_revenue.csv             # Daily aggregated revenue
│── forecast_output.csv           # Forecast results
│── product_summary.csv           # Product-level analytics
│── retail_summary.xlsx           # KPI exports
│── visuals/                      # Project plots and charts
│── README.md                     # Project documentation
________________________________________
🧼 Data Cleaning & Preprocessing
•	Removed negative quantities and prices.
•	Dropped missing Customer ID values.
•	Created a new Revenue column (Quantity × Price).
•	Standardized date formats.
•	Automated cleaning workflow using a reusable function.
________________________________________
📈 Exploratory Data Analysis (EDA)
Key EDA steps include: - Daily, monthly, and hourly revenue trends. - Seasonality patterns (weekly & yearly). - Heatmaps for daily activity and purchasing intensity. - Product performance (top sellers, contribution charts). - Customer analysis using RFM segmentation. - Outlier identification using Z-score & IQR. - Country-wise revenue distribution.
________________________________________
📊 Statistical Analysis
•	T-tests for weekday vs weekend buying behavior.
•	ANOVA to compare revenue across countries.
•	Chi-square tests for product-country associations.
•	Skewness & kurtosis for distribution understanding.
•	Seasonal decomposition (trend, seasonal, residual).
________________________________________
🧠 Feature Engineering
•	Date-based features: weekday, month, year.
•	Lag features: 1-day, 7-day, 30-day revenue.
•	Rolling windows for trend smoothing.
•	Variability indicators (rolling std).
________________________________________
🔮 Forecasting Models
1. Prophet
•	Captures weekly & yearly seasonality.
•	Good for business trend visualization.
2. SARIMAX
•	Strong for capturing autoregressive patterns.
•	Effective weekly seasonal modeling.
3. LightGBM (best performing)
•	Feature-driven forecasting.
•	Captures non-linear patterns.
Metrics Used: MAPE, RMSE, MAE
________________________________________
💡 Key Business Insights
•	Revenue peaks heavily during end-of-year holiday seasons.
•	UK dominates total revenue share.
•	The top 20% products generate almost 80% of total revenue.
•	Customer base is long-tailed, with a small segment contributing major revenue.
•	Revenue shows strong weekly seasonality useful for forecasting.
________________________________________
📁 Tools & Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, Statsmodels, Prophet, LightGBM, Scikit-learn, SQL/SQLite, Power BI/Tableau, Git.
________________________________________
📌 Future Improvements
•	Add hyperparameter tuning for LightGBM.
•	Build a full Power BI dashboard.
•	Implement LSTM/Neural forecasting models.
•	Add automated pipeline using Airflow.
________________________________________

