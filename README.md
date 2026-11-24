# Retail Customer Satisfaction Prediction

This project predicts whether a customer will be unhappy with a purchase. I used a retail dataset with around 300,000 transactions, trained a few machine learning models, and deployed the best one as a Streamlit web app.

I followed the CRISP-DM process, so the sections below go through each of its six steps.

**Live app:** _add your Streamlit link here_

---

## 1. Business Understanding

When a customer has a bad experience, the store usually only finds out after the bad review is already written. By then it's often too late to keep them.

The idea here is to predict dissatisfaction from the order details, so the business can reach out early with a discount, a follow-up call or better support.

The question I'm answering: **based on who the customer is and what they bought, will they be dissatisfied?**

I turned the `Feedback` column into the target:
- **Bad** or **Average** → Dissatisfied (1)
- **Good** or **Excellent** → Satisfied (0)

For the business, the main thing is to catch as many unhappy customers as possible, so I focused on **recall** for the dissatisfied class.

---

## 2. Data Understanding

The dataset (`new_retail_data.csv`) has about 302,000 rows and 30 columns, covering March 2023 to February 2024 across five countries (USA, UK, Germany, Canada and Australia).

It includes customer info (age, gender, income, segment, country), transaction details (date, quantity, amount), product info (category, brand), order info (shipping, payment, status) and customer feedback.

What I noticed while exploring:
- Only a small number of values are missing, a few hundred per column.
- About 65% of customers were satisfied and 35% weren't, so the classes are a bit imbalanced.
- The `Ratings` column is basically the same thing as `Feedback` (a rating of 1 always means Bad, and so on). Using it would be cheating, so I left it out.
- The same Customer_ID sometimes shows up with different ages and countries, so I treated each row as a separate transaction instead of grouping by customer.
- Premium customers and people buying electronics or groceries tend to be happier. Regular customers were the least satisfied.

---

## 3. Data Preparation

- Created the `Dissatisfied` column and removed rows with no feedback
- Dropped personal columns (name, email, phone, address) and the `Ratings` column
- Kept 13 features: Age, Gender, Income, Customer_Segment, Country, Month, Total_Purchases, Amount, Total_Amount, Product_Category, Shipping_Method, Payment_Method and Order_Status
- Filled missing numbers with the median and missing categories with the most common value
- One-hot encoded the categorical columns and scaled the numeric ones
- Split the data into 80% training and 20% testing

---

## 4. Modeling

I tried four models:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

Because of the class imbalance, I used `class_weight='balanced'` so the models wouldn't just predict "satisfied" for everyone.

Random Forest did the best, so I checked it with 10-fold cross-validation (the scores were stable, with no overfitting) and then tuned it with GridSearchCV. The final model is a Random Forest with 100 trees and a max depth of 12.

Everything (cleaning, encoding, scaling and the model) is wrapped in a single scikit-learn pipeline, so the app can take raw input directly.

---

## 5. Evaluation

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | 0.60 | 0.59 |
| KNN | 0.62 | 0.56 |
| Decision Tree | 0.61 | 0.60 |
| Random Forest | 0.63 | 0.62 |
| **Tuned Random Forest** | **0.63** | **0.63** |

The final model catches about **69% of dissatisfied customers**, which was the main goal. It does flag some happy customers by mistake (precision is around 0.48 for the dissatisfied class), but for this use case that's fine: sending a discount to a happy customer costs much less than losing an unhappy one.

The results are moderate. The dataset doesn't have things like delivery delays or complaint history, which would probably matter a lot more for satisfaction. With features like those, the model could do much better.

---

## 6. Deployment

I saved the trained pipeline with `joblib` and built a simple Streamlit app around it. You fill in the customer and order details, click Predict, and the app tells you whether the customer is likely to be satisfied or not, along with the probability.

The app is hosted on Streamlit Community Cloud straight from this repo.

**Files in this repo:**
- `app.py`: the Streamlit app
- `satisfaction_model.pkl`: the trained model
- `requirements.txt`: the libraries needed
- `README.md`: this file

**To run it on your own computer:**
```bash
git clone https://github.com/Ahmedalyafei/retail-satisfaction-app.git
cd retail-satisfaction-app
pip install -r requirements.txt
streamlit run app.py
```

---

## What I'd improve next

- Add more useful data, like delivery time, returns and support tickets
- Try other models such as XGBoost or LightGBM
- Adjust the prediction threshold depending on what the business cares about more
- Retrain the model regularly as new data comes in

---

**Tools used:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit

**Made by Ahmed**
