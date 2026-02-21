import kagglehub


path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Path to dataset files:", path)
import os
import pandas as pd
import seaborn as sns
import numpy as np

# Load datasets
customers = pd.read_csv(os.path.join(path, 'olist_customers_dataset.csv'))
items = pd.read_csv(os.path.join(path, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(path, 'olist_order_payments_dataset.csv'))
reviews = pd.read_csv(os.path.join(path, 'olist_order_reviews_dataset.csv'))
orders = pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv'))
products = pd.read_csv(os.path.join(path, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(path, 'olist_sellers_dataset.csv'))
category_translation = pd.read_csv(os.path.join(path, 'product_category_name_translation.csv'))
geolocation  = pd.read_csv(os.path.join(path, 'olist_geolocation_dataset.csv'))