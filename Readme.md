# Olist Brazilian Marketplace Analysis API

This project provides a FastAPI application for predicting order delivery delays and categorizing sellers based on their behavior and revenue.

## Problem Statement

"We're growing fast but our margins are getting squeezed. Sellers are complaining, buyers leave bad reviews, and we don't really know where to focus. We've heard ML can help but we don't know what to build first. Can you look at our data and tell us what would actually make a difference?"

## Getting Started

### Option 1: Docker Setup

The easiest way to run the application is using Docker. This will automatically install dependencies, train the required models, and start the FastAPI server.

1. Build the Docker image:
   ```bash
   docker build -t api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 api
   ```

3. The API will now be accessible at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

### Option 2: Manual Setup

To run the application manually on your local machine, follow these steps:

1. Ensure you have Python 3.10+ installed.
    ```bash
    python --version
    ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the required dependencies:
   ```bash
   cd src
   ```

4. Start the FastAPI application:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. The API will now be accessible at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

---


## Delayed Delivery Prediction Model

This project also includes a `delayed_delivery_regression` model that predicts how delayed an order is expected to be.

### Features Used
The model is trained on a comprehensive set of features, including:
- order_size
- customer_seller_distance_km
- delivery_delay_days
- purchase_month
- purchase_day_of_week
- purchase_hour
- is_weekend
- purchase_quarter
- customer_city_avg_delay
- customer_zip_avg_delay
- customer_zip_order_volume
- seller_zip_avg_delay
- customer_state_encoded
- seller_state_encoded
- state_pair_encoded
- customer_region_encoded
- seller_region_encoded
- region_pair_encoded

### Why Build This?

I analyzed the massive impact of late deliveries on customer satisfaction:
- Late orders: 7,183 (7.5%)
- Average review score (on-time): 4.29
- Average review score (late): 2.46
- Review score drop when late: 1.83 points

If we could prevent just 50% of late orders:
→ 3,591 orders would have better reviews
→ Overall average review score would increase by 0.068 points

Approximately 33% of the bad review scores are caused directly by delayed deliveries. If we can predict in advance that an order is going to be delayed, we can proactively reach out to customers with apologies, coupons, or discounts to offset the poor experience and preserve our review score.

---


## Seller Behaviour Clustering & Pareto Analysis

This project is my attempt to understand how sellers behave in a marketplace — not just who earns the most, but how they operate.

Instead of doing simple revenue ranking, I used clustering to uncover hidden seller segments.

### What I Did

I built two types of clustering models:

#### Scale-Based Clustering (Revenue Focused)
```markdown
I clustered sellers using a comprehensive feature set:
- Total revenue
- Total orders
- Average order value
- Average delivery time
- Late delivery rate
- Average freight cost per order
- Average processing time
- Average review score
- Review count

In this model, total revenue and total orders are the dominant features, which naturally segments sellers by their scale. This helped me validate the Pareto Principle (80/20 rule) — a small group of sellers contributes a disproportionately large share of total revenue.

**For this scale-based clustering (k=2), the model achieved a silhouette score of 72%.**

This clearly distinguishes:
- A small “high-impact” seller group driving GMV (this group is small in number but forms ~50% of the company revenue).
- A larger group of emerging or smaller sellers.
```

#### Behaviour-Based Clustering (Revenue Independent) To Validate A Seller can be small but operationally excellent or high-revenue but operationally risky
```markdown
I removed revenue and order count completely, clustering sellers only on operational behavior:
- Delivery speed
- Late delivery rate
- Freight cost per order
- Processing time
- Review score

This revealed something interesting:

**For this model, I got a silhouette score of approximately 60% for k=3. Even though I got a better score for k=2, I chose k=3 because it helped me discover more nuanced, actionable operational profiles rather than a simple 'good vs bad' binary split.**

**A seller can be:**
- Small but operationally excellent
- High-revenue but operationally risky

That’s powerful for marketplace strategy.
```

---
