# Step 1 => Import required libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Step 2 => Load all datasets
def load_datasets():
    customers = pd.read_csv("./Dependencies/data/olist_customers_dataset.csv")
    order_items = pd.read_csv("./Dependencies/data/olist_order_items_dataset.csv")
    order_payments = pd.read_csv("./Dependencies/data/olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv("./Dependencies/data/olist_order_reviews_dataset.csv")
    orders = pd.read_csv("./Dependencies/data/olist_orders_dataset.csv")
    products = pd.read_csv("./Dependencies/data/olist_products_dataset.csv")
    sellers = pd.read_csv("./Dependencies/data/olist_sellers_dataset.csv")
    category_translation = pd.read_csv("./Dependencies/data/product_category_name_translation.csv")
    return customers, order_items, order_payments, order_reviews, orders, products, sellers, category_translation

# Step 3 => Merge datasets
def merge_datasets(orders, customers, order_items, products, order_payments, order_reviews, sellers, category_translation):
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(order_items, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(order_payments, on="order_id", how="left")
    df = df.merge(order_reviews[['order_id', 'review_score']], on="order_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(category_translation, on="product_category_name", how="left")
    df["total_price"] = df["price"] * df["order_item_id"]
    df["total_freight"] = df["freight_value"] * df["order_item_id"]
    df["total_payment"] = df["payment_value"]
    df["avg_review_score"] = df["review_score"]
    return df

# Step 4 => Preprocess data
def preprocess_data(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors='coerce')
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors='coerce')
    df["order_completion_time"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df.dropna(subset=["order_completion_time"], inplace=True)
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = imputer.fit_transform(df_numeric)
    categorical_cols = ["customer_state", "seller_state", "product_category_name_english", "payment_type"]
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    for col in existing_categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    numerical_cols = ["total_price", "total_freight", "total_payment", "avg_review_score", "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday"]
    existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
    if existing_numerical_cols:
        scaler = StandardScaler()
        df[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])
    else:
        print("No numerical features available for scaling!")
    return df

# Step 5 => Optimize XGBoost Model using GridSearchCV
def optimize_model(X_train, y_train):
    param_grid = {
        'n_estimators': [500, 700],
        'learning_rate': [0.03, 0.05],
        'max_depth': [6, 8],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'random_state': [42]
    }
    xgb = XGBRegressor()
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Step 6 => Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    plt.figure(figsize=(10,6))
    sns.histplot(y_test - y_pred, kde=True, bins=30)
    plt.xlabel("Residual Errors")
    plt.title("Residual Distribution")
    plt.show()
    feature_importance = model.feature_importances_
    labels = X_test.columns
    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importance, y=labels, palette="coolwarm")
    plt.title("Feature Importance in Model")
    plt.show()
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    return r2

# Step 7 => Main Execution
def main():
    customers, order_items, order_payments, order_reviews, orders, products, sellers, category_translation = load_datasets()
    df = merge_datasets(orders, customers, order_items, products, order_payments, order_reviews, sellers, category_translation)
    df = preprocess_data(df)
    features = ["total_price", "total_freight", "total_payment", "avg_review_score", "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", "customer_state", "seller_state", "product_category_name_english", "payment_type"]
    existing_features = [col for col in features if col in df.columns]
    if not existing_features:
        print("Error: No valid features available for training!")
        return
    X = df[existing_features]
    y = df["order_completion_time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = optimize_model(X_train, y_train)
    joblib.dump(best_model, "xgboost_optimized.pkl")
    print("Model saved as 'xgboost_optimized.pkl'!")
    r2 = evaluate_model(best_model, X_test, y_test)
    if r2 > 0.85:
        print("Model is performing exceptionally well with high accuracy!")

if __name__ == "__main__":
    main()
    
