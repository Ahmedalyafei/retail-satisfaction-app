
import streamlit as st
import joblib
import pandas as pd

classifier = joblib.load('satisfaction_model.pkl')

def predict_dissatisfaction(Age, Gender, Income, Customer_Segment, Country, Month,
                            Total_Purchases, Amount, Product_Category,
                            Shipping_Method, Payment_Method, Order_Status):
    input_data = pd.DataFrame([{
        'Age': Age,
        'Gender': Gender,
        'Income': Income,
        'Customer_Segment': Customer_Segment,
        'Country': Country,
        'Month': Month,
        'Total_Purchases': Total_Purchases,
        'Amount': Amount,
        'Total_Amount': Total_Purchases * Amount,
        'Product_Category': Product_Category,
        'Shipping_Method': Shipping_Method,
        'Payment_Method': Payment_Method,
        'Order_Status': Order_Status
    }])
    prediction = classifier.predict(input_data)
    proba = classifier.predict_proba(input_data)[0][1]
    label = 'Customer Will Be Dissatisfied' if prediction[0] == 1 else 'Customer Will Be Satisfied'
    return label, proba

def main():
    st.title("Customer Satisfaction Prediction")
    html_temp = """
        <div style="background-color:red;padding:10px">
            <h2 style="color:white;text-align:center;">This our Streamlit Deployment App</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("Enter transaction details to predict whether the customer will be dissatisfied.")

    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Income = st.selectbox("Income", ['Low', 'Medium', 'High'])
    Customer_Segment = st.selectbox("Customer Segment", ['New', 'Regular', 'Premium'])
    Country = st.selectbox("Country", ['USA', 'UK', 'Germany', 'Canada', 'Australia'])
    Month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December'])
    Total_Purchases = st.number_input("Total Purchases (items)", min_value=1, max_value=10, value=3)
    Amount = st.number_input("Amount per item", min_value=0.0, value=250.0, format="%.2f")
    Product_Category = st.selectbox("Product Category", ['Electronics', 'Grocery', 'Clothing', 'Books', 'Home Decor'])
    Shipping_Method = st.selectbox("Shipping Method", ['Same-Day', 'Express', 'Standard'])
    Payment_Method = st.selectbox("Payment Method", ['Credit Card', 'Debit Card', 'Cash', 'PayPal'])
    Order_Status = st.selectbox("Order Status", ['Delivered', 'Shipped', 'Processing', 'Pending'])

    st.write(f"Total Amount: {Total_Purchases * Amount:,.2f}")

    if st.button("Predict"):
        result, proba = predict_dissatisfaction(
            Age, Gender, Income, Customer_Segment, Country, Month,
            Total_Purchases, Amount, Product_Category,
            Shipping_Method, Payment_Method, Order_Status
        )
        st.success(f"Prediction: {result}")
        st.write(f"Probability of dissatisfaction: {proba:.1%}")

if __name__ == '__main__':
    main()
