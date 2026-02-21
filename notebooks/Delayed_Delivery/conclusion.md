# Conclusion: Delivery Delay Prediction Model

## Overview
This document summarizes the findings from the **Delayed Delivery Prediction** analysis. The model was developed to predict the number of days an order is delayed relative to its initial estimate, which is a critical factor for customer satisfaction on the Olist platform.

## 1. Model Performance & Metrics
A **Random Forest Regressor** was trained and evaluated on the dataset. The key performance indicators are:

*   **Mean Absolute Error (MAE):** **5.57 days**. On average, predictions deviate from the actual delay by approximately 5.6 days.
*   **Root Mean Squared Error (RMSE):** **8.43 days**. The gap between MAE and RMSE indicates that the model is sensitive to large outliers (orders with extreme delays).
*   **R² Score (Coefficient of Determination):** **0.3102**. The model explains about **31% of the variance** in delivery delays. This suggests that while the model captures systemic trends, logistics remain highly stochastic.

## 2. Key Features Influencing Delays
The model's predictive power is driven by several engineered features:

*   **Haversine Distance:** The calculated physical distance (in km) between the `customer_zip_code` and `seller_zip_code` proved to be a primary predictor.
*   **Historical Aggregates:** Average historical delays calculated at the **Zip Code** and **City** levels provided context for logistical "bottlenecks."
*   **Temporal Features:** Purchase timing (month, hour, weekend) accounts for fluctuations in order volume and carrier capacity.
*   **Order Size:** The number of items in a single order correlates with increased complexity in processing and shipping.

## 3. Business Insights
*   **Impact on Reviews:** Orders that are late (approximately 7.5% of total) see their average review score drop from **4.29** (on-time) to **2.46** (late).
*   **Late Order Severity:** Among late orders, the average delay is **10.02 days**, indicating that once an order misses its window, the delay tends to be significant.
*   **Risk Mitigation:** The model can be used to identify "High Risk" orders at the time of purchase, allowing Olist to adjust customer expectations or expedite carrier handovers.

## 4. Model Limitations
*   **Outlier Influence:** The model struggles to predict the rare but extreme cases (delay > 30 days) which significantly impact the RMSE.
