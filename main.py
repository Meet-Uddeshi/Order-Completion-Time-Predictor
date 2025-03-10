# Step => 1 Importing requirement libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step => 2 Function to load all CSV datasets
def load_datasets():
    customers = pd.read_csv("./Dependencies./data./olist_customers_dataset.csv")
    geolocation = pd.read_csv("./Dependencies./data./olist_geolocation_dataset.csv")
    order_items = pd.read_csv("./Dependencies./data./olist_order_items_dataset.csv")
    order_payments = pd.read_csv("./Dependencies./data./olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv("./Dependencies./data./olist_order_reviews_dataset.csv")
    orders = pd.read_csv("./Dependencies./data./olist_orders_dataset.csv")
    products = pd.read_csv("./Dependencies./data./olist_products_dataset.csv")
    sellers = pd.read_csv("./Dependencies./data./olist_sellers_dataset.csv")
    category_translation = pd.read_csv("./Dependencies./data./product_category_name_translation.csv")
    return customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers, category_translation

# Step => 3 Function to merge multiple datasets into a single dataframe
def merge_datasets(orders, customers, order_items, products, order_payments, order_reviews, sellers, category_translation):
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(order_items, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(order_payments, on="order_id", how="left")
    df = df.merge(order_reviews[['order_id', 'review_score']], on="order_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(category_translation, on="product_category_name", how="left")
    return df

# Step => 4 Function to preprocess data by handling missing values, encoding categorical variables, and normalizing numerical features
def preprocess_data(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors='coerce')
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors='coerce')
    df["order_completion_time"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df.dropna(subset=["order_completion_time"], inplace=True)
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)
    df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)
    categorical_cols = ["customer_state", "seller_state", "product_category_name_english", "payment_type"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    scaler = StandardScaler()
    numerical_cols = ["total_price", "total_freight", "total_payment", "avg_review_score", "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# Step => 5 Function to train the GradientBoostingRegressor model using GridSearchCV
def train_model(X_train, y_train):
    tuned_parameters = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    gbr = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gbr, tuned_parameters, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search

# Step => 6 Function to evaluate model performance and generate visualizations
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    plt.figure(figsize=(10,6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha":0.5})
    plt.xlabel("Actual Order Completion Time")
    plt.ylabel("Predicted Order Completion Time")
    plt.title("Actual vs Predicted Order Completion Time")
    plt.show()
    feature_importance = model.feature_importances_
    labels = X_test.columns
    plt.figure(figsize=(10,6))
    plt.pie(feature_importance, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Feature Importance in Model")
    plt.show()
    metrics = [mae, mse, rmse, r2]
    names = ["MAE", "MSE", "RMSE", "R²"]
    plt.figure(figsize=(8,6))
    sns.barplot(x=names, y=metrics, palette="viridis")
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.show()
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R² Score: {r2}")
    return r2

# Step => 7 Main function to execute the full process
def main():
    customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers, category_translation = load_datasets()
    df = merge_datasets(orders, customers, order_items, products, order_payments, order_reviews, sellers, category_translation)
    df = preprocess_data(df)
    features = ["num_items", "total_price", "total_freight", "total_payment", "avg_review_score", "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", "customer_state", "seller_state", "product_category_name_english", "payment_type"]
    X = df[features]
    y = df["order_completion_time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = train_model(X_train, y_train).best_estimator_
    joblib.dump(best_model, "gradient_boosting_model.pkl")
    print("Model saved as 'gradient_boosting_model.pkl'!")
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
