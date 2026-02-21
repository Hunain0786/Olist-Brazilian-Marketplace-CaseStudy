curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_state": "SP",
  "customer_city": "sao paulo",
  "customer_zip_code_prefix": 1046,
  "seller_state": "MG",
  "seller_city": "belo horizonte",
  "seller_zip_code_prefix": 30130,
  "order_size": 2,
  "order_purchase_timestamp": "2024-03-15T14:30:00"
}'


curl -X 'POST' \
  'http://localhost:8000/predict_revenue_category' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "total_revenue": 15000.50,
  "total_orders": 120,
  "avg_order_value": 125.0,
  "avg_delivery_time_days": 4.5,
  "late_delivery_rate": 0.05,
  "avg_freight_per_order": 15.2,
  "avg_processing_time_days": 1.2,
  "avg_review_score": 4.8,
  "review_count": 95
}'


curl -X 'POST' \
  'http://localhost:8000/predict_behavior_category' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "avg_order_value": 85.5,
  "avg_delivery_time_days": 6.2,
  "late_delivery_rate": 0.12,
  "avg_freight_per_order": 18.5,
  "avg_processing_time_days": 2.5,
  "avg_review_score": 3.9
}'

