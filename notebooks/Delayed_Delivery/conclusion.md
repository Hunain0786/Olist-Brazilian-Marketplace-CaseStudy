# Conclusion: Delivery Delay Prediction Model

## Overview
This document summarizes the findings from the **Delayed Delivery Prediction** analysis. The model was developed to predict the number of days an order is delayed relative to its initial estimate, which is a critical factor for customer satisfaction on the Olist platform.

## 1. Motivation: Bad Review Analysis
The primary driver for this model was understanding the correlation between late deliveries and negative customer feedback. Exploratory analysis revealed:
*   **Total bad reviews (1-2 stars):** 12,347
*   **Bad reviews associated with late delivery:** 4,096
*   **Direct Attribution:** **33.17% of all bad reviews** are caused directly by delivery delays.
*   **Score Disparity:** Customers who receive orders on-time/early give an average score of **4.29**, whereas late orders drop to **2.46**.

## 2. Evaluation Metric Rationale
To evaluate the regression model's accuracy, we utilized three distinct metrics:

*   **Mean Absolute Error (MAE):** We chose MAE because it provides an intuitive "real-world" error in units of days. An MAE of 5.57 tells us exactly how much our predictions deviate from reality on average.
*   **Root Mean Squared Error (RMSE):** Logistics delays are non-linear in their impact; a 10-day delay is significantly more damaging to brand trust than two 5-day delays. RMSE penalizes these larger errors more heavily, reflecting the high business cost of extreme delays.
*   **R² Score:** This metric was used to gauge how much of the "logistical noise" the model could actually explain. Given the unpredictable nature of transit (traffic, weather, etc.), an R² of 0.31 indicates the model has successfully captured a significant portion of the underlying patterns.

---

## 3. Model Performance & Metrics
A **Random Forest Regressor** was trained and evaluated on the dataset. The key performance indicators are:

*   **Mean Absolute Error (MAE):** **5.57 days**.
*   **Root Mean Squared Error (RMSE):** **8.43 days**.
*   **R² Score:** **0.3102**. The model explains about **31% of the variance** in delivery delays.

## 3. Key Features Influencing Delays
The model's predictive power is driven by:
*   **Haversine Distance:** Physical distance between customer and seller.
*   **Historical Aggregates:** Average historical delays by City and Zip Code.
*   **Temporal Features:** Purchase timing (month, hour, weekend).
*   **Order Size:** Number of items in the order.

## 4. Business Impact Analysis
Using this model to flag and prevent delays has a direct quantifiable impact on platform health:
*   **Late orders:** 7,183 (7.5% of total orders).
*   **Review score drop:** A late delivery costs the platform **1.83 points** per review on average.
*   **Potential Gain:** If we could prevent just **50% of late orders** (3,591 orders):
    *   Overall average review score would increase by **0.068 points**.
    *   Significant reduction in customer support tickets related to 1-2 star reviews.

## 5. Model Limitations
*   **Data Gaps:** Lack of real-time transit data (weather, carrier strikes).
*   **Non-Linear Delays:** Stochastic nature of logistics.
*   **Outlier Influence:** Difficulty in predicting extreme "Black Swan" delay events.
