import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step => 1 Function to load all datasets
def load_datasets():
    customers = pd.read_csv("./Dependencies/data/olist_customers_dataset.csv")
    geolocation = pd.read_csv("./Dependencies/data/olist_geolocation_dataset.csv")
    order_items = pd.read_csv("./Dependencies/data/olist_order_items_dataset.csv")
    order_payments = pd.read_csv("./Dependencies/data/olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv("./Dependencies/data/olist_order_reviews_dataset.csv")
    orders = pd.read_csv("./Dependencies/data/olist_orders_dataset.csv")
    products = pd.read_csv("./Dependencies/data/olist_products_dataset.csv")
    sellers = pd.read_csv("./Dependencies/data/olist_sellers_dataset.csv")
    category_translation = pd.read_csv("./Dependencies/data/product_category_name_translation.csv")
    return customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers, category_translation

# Step => 2 Function to merge multiple datasets into a single dataframe
def merge_datasets(orders, customers, order_items, products, order_payments, order_reviews, sellers, category_translation):
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(order_items, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(order_payments, on="order_id", how="left")
    df = df.merge(order_reviews[['order_id', 'review_score']], on="order_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(category_translation, on="product_category_name", how="left")
    return df

# Step => 3 Function to preprocess data by handling missing values, encoding categorical variables, and normalizing numerical features
def preprocess_data(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors='coerce')
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors='coerce')
    df["order_completion_time"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600  # Convert to hours
    df.dropna(subset=["order_completion_time"], inplace=True)
    
    # Step => 3.1 Feature Engineering
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    
    # Step => 3.2 Handling missing values using KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = imputer.fit_transform(df_numeric)
    
    # Step => 3.3 Encoding categorical variables
    categorical_cols = ["customer_state", "seller_state", "product_category_name_english", "payment_type"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Step => 3.4 Feature Scaling using StandardScaler
    scaler = StandardScaler()
    numerical_cols = ["total_price", "total_freight", "total_payment", "avg_review_score", "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  
    return df

# Step => 4 Function to train the XGBoost model for better accuracy
def train_model(X_train, y_train):
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

# Step => 5 Function to evaluate model performance and visualize results
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Step => 5.1 Residual Plot for error distribution
    plt.figure(figsize=(10,6))
    sns.histplot(y_test - y_pred, kde=True, bins=30)
    plt.xlabel("Residual Errors")
    plt.title("Residual Distribution")
    plt.show()
    
    # Step => 5.2 Feature Importance Plot
    feature_importance = model.feature_importances_
    labels = X_test.columns
    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importance, y=labels, palette="coolwarm")
    plt.title("Feature Importance in Model")
    plt.show()
    
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")
    return r2

# Step => 6 Main function to execute the full process
def main():
    customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers, category_translation = load_datasets()
    df = merge_datasets(orders, customers, order_items, products, order_payments, order_reviews, sellers, category_translation)
    df = preprocess_data(df)
    features = ["total_price", "total_freight", "total_payment", "avg_review_score", "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", "customer_state", "seller_state", "product_category_name_english", "payment_type"]
    X = df[features]
    y = df["order_completion_time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = train_model(X_train, y_train)
    joblib.dump(best_model, "xgboost_model.pkl")
    print("Model saved as 'xgboost_model.pkl'!")
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
