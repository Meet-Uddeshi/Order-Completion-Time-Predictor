# Step 1 => Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Step  2 =>Load the trained model
def load_model():
    """Load the pre-trained XGBoost model from a file."""
    return joblib.load("xgboost_optimized.pkl")

# Step 3 => Define the prediction function
def predict_delivery_time(model, total_price, total_freight, total_payment, avg_review_score, 
                          purchase_hour, purchase_day, purchase_month, purchase_weekday, 
                          customer_state, seller_state, product_category, payment_type):
    """Prepare input data and predict the estimated delivery time."""
    input_data = pd.DataFrame([[
        total_price, total_freight, total_payment, avg_review_score, 
        purchase_hour, purchase_day, purchase_month, purchase_weekday, 
        customer_state, seller_state, product_category, payment_type
    ]], columns=["total_price", "total_freight", "total_payment", "avg_review_score", 
                 "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", 
                 "customer_state", "seller_state", "product_category_name_english", "payment_type"])    
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)  

# Step 4 => Define the UI for the app
def setup_ui():
    """Initialize and set up the Streamlit UI."""
    st.set_page_config(page_title="Order Delivery Time Prediction", layout="centered")
    st.title("🚚 Order Delivery Time Prediction")
    st.write("Predict the estimated delivery time based on order details.")
    st.sidebar.header("📌 How to Use")
    st.sidebar.write("""
    - Enter order details including product category, customer location, and shipping method.
    - Click **'Predict Delivery Time'** to get an estimated time.
    - The model will predict the time in **hours**.
    """)
    
# Step 5 => Define the function to get user inputs
def get_user_inputs():
    """Create user input fields for order details and return the input values."""
    st.subheader("📝 Enter Order Details")
    col1, col2 = st.columns(2)
    with col1:
        total_price = st.number_input("💰 Total Price (in currency)", min_value=0.0, step=1.0, format="%.2f")
        total_freight = st.number_input("🚚 Total Freight Cost", min_value=0.0, step=1.0, format="%.2f")
        total_payment = st.number_input("💳 Total Payment Amount", min_value=0.0, step=1.0, format="%.2f")
        avg_review_score = st.slider("⭐ Average Review Score (1-5)", min_value=1.0, max_value=5.0, step=0.1)

    with col2:
        purchase_hour = st.slider("⏰ Purchase Hour (0-23)", min_value=0, max_value=23, step=1)
        purchase_day = st.slider("📅 Purchase Day (1-31)", min_value=1, max_value=31, step=1)
        purchase_month = st.slider("📆 Purchase Month (1-12)", min_value=1, max_value=12, step=1)
        purchase_weekday = st.slider("📊 Purchase Weekday (0=Monday, 6=Sunday)", min_value=0, max_value=6, step=1)
    st.subheader("📍 Order & Shipping Details")
    customer_state = st.selectbox("📍 Customer State", list(range(27)), format_func=lambda x: f"State {x}")
    seller_state = st.selectbox("🏪 Seller State", list(range(27)), format_func=lambda x: f"State {x}")
    product_category = st.selectbox("📦 Product Category", list(range(50)), format_func=lambda x: f"Category {x}")
    payment_type = st.selectbox("💲 Payment Type", list(range(4)), format_func=lambda x: f"Type {x}")
    return (total_price, total_freight, total_payment, avg_review_score, 
            purchase_hour, purchase_day, purchase_month, purchase_weekday, 
            customer_state, seller_state, product_category, payment_type)

# Step 6 => Define the function to display results
def display_results(model):
    """Handle the prediction process and display the results."""

    inputs = get_user_inputs()
    if st.button("📊 Predict Delivery Time"):
        result = predict_delivery_time(model, *inputs)
        st.success(f"⏳ Estimated Delivery Time: **{result} hours**")
        st.info("This prediction is based on historical data and may vary.")

# Step 7 => Define the main function
def main():
    """Main function to run the Streamlit app."""
    model = load_model()
    setup_ui()
    display_results(model)
    st.write("---")
    st.write("📌 Order Delivering Time Smarter Predictions!**")

# Step 8 => Run the app
if __name__ == "__main__":
    main()
