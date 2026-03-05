# API Testing — curl Commands

Live server: **`http://13.60.69.76:8000`**
Interactive docs: [http://13.60.69.76:8000/docs)

---

## 1. Predict Seller Revenue Category — `/predict_revenue_category`

**Bash / Git Bash / WSL:**
```bash
curl -X POST http://13.60.69.76:8000/predict_revenue_category \
  -H "Content-Type: application/json" \
  -d '{"total_revenue":15000.50,"total_orders":120,"avg_order_value":125.0,"avg_delivery_time_days":4.5,"late_delivery_rate":0.05,"avg_freight_per_order":15.2,"avg_processing_time_days":1.2,"avg_review_score":4.8,"review_count":95}'
```

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://13.60.69.76:8000/predict_revenue_category" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"total_revenue":15000.50,"total_orders":120,"avg_order_value":125.0,"avg_delivery_time_days":4.5,"late_delivery_rate":0.05,"avg_freight_per_order":15.2,"avg_processing_time_days":1.2,"avg_review_score":4.8,"review_count":95}' `
  -UseBasicParsing | Select-Object -ExpandProperty Content
```

---

## 2. Predict Seller Behavior Category — `/predict_behavior_category`

**Bash / Git Bash / WSL:**
```bash
curl -X POST http://13.60.69.76:8000/predict_behavior_category \
  -H "Content-Type: application/json" \
  -d '{"avg_order_value":85.5,"avg_delivery_time_days":6.2,"late_delivery_rate":0.12,"avg_freight_per_order":18.5,"avg_processing_time_days":2.5,"avg_review_score":3.9}'
```

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://13.60.69.76:8000/predict_behavior_category" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"avg_order_value":85.5,"avg_delivery_time_days":6.2,"late_delivery_rate":0.12,"avg_freight_per_order":18.5,"avg_processing_time_days":2.5,"avg_review_score":3.9}' `
  -UseBasicParsing | Select-Object -ExpandProperty Content
```

---

## 3. Predict Delivery Delay — `/predict`

**Bash / Git Bash / WSL:**
```bash
curl -X POST http://13.60.69.76:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "seller_id": "3442f8959a84dea7ee197c632cb2df15",
    "customer_state": "SP",
    "customer_city": "sao paulo",
    "seller_state": "MG",
    "seller_city": "belo horizonte",
    "product_category_name_english": "health_beauty",
    "product_weight_g": 500,
    "product_height_cm": 16,
    "product_width_cm": 11,
    "product_length_cm": 18,
    "total_order_items": 2,
    "freight_value": 15.1,
    "price": 59.9,
    "order_purchase_timestamp": "2024-03-15T14:30:00",
    "order_estimated_delivery_date": "2024-03-28T00:00:00"
  }'

'
```

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://13.60.69.76:8000/predict" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"seller_id":"3442f8959a84dea7ee197c632cb2df15","customer_state":"SP","customer_city":"sao paulo","seller_state":"MG","seller_city":"belo horizonte","product_category_name_english":"health_beauty","product_weight_g":500,"product_height_cm":16,"product_width_cm":11,"product_length_cm":18,"total_order_items":2,"freight_value":15.1,"price":59.9,"order_purchase_timestamp":"2024-03-15T14:30:00","order_estimated_delivery_date":"2024-03-28T00:00:00"}' `
  -UseBasicParsing | Select-Object -ExpandProperty Content
```

---

> **Note:** The backtick `` ` `` is PowerShell's line-continuation character (equivalent to `\` in bash).
