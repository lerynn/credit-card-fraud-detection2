import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# Load saved model files
# -----------------------------
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")

except FileNotFoundError:
    st.error("One or more model files could not be found.")
    st.stop()

except Exception as error:
    st.error(f"Error loading model files: {error}")
    st.stop()


# -----------------------------
# Page title and description
# -----------------------------
st.title("💳 FraudGuard AI")
st.subheader("Credit Card Fraud Detection")

st.write(
    "Enter the transaction information below. "
    "The machine learning model will estimate whether the transaction "
    "is likely to be legitimate or fraudulent."
)

st.warning(
    "This application is created for educational purposes only."
)


# -----------------------------
# Dropdown options
# -----------------------------
category_options = list(
    label_encoders["category"].classes_
)

state_options = list(
    label_encoders["state"].classes_
)

gender_options = [
    "Female",
    "Male"
]


# -----------------------------
# Input form
# -----------------------------
with st.form("fraud_prediction_form"):

    st.markdown("### Transaction Information")

    column1, column2 = st.columns(2)

    with column1:

        category = st.selectbox(
            "Merchant Category",
            category_options
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        gender = st.selectbox(
            "Customer Gender",
            gender_options
        )

        state = st.selectbox(
            "Customer State",
            state_options
        )

        city_population = st.number_input(
            "City Population",
            min_value=0,
            value=50000,
            step=1000
        )

    with column2:

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        transaction_day = st.selectbox(
            "Transaction Day",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda day: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][day]
        )

        transaction_month = st.selectbox(
            "Transaction Month",
            options=list(range(1, 13)),
            format_func=lambda month: [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            ][month - 1]
        )

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

    predict_button = st.form_submit_button(
        "Predict Transaction"
    )


# -----------------------------
# Make prediction
# -----------------------------
if predict_button:

    try:

        # Convert category using saved encoder
        encoded_category = label_encoders[
            "category"
        ].transform(
            [category]
        )[0]

        # Convert Female and Male back to F and M
        gender_mapping = {
            "Female": 0,
            "Male": 1
        }

        encoded_gender = gender_mapping[gender]

        # Convert state using saved encoder
        encoded_state = label_encoders[
            "state"
        ].transform(
            [state]
        )[0]

        # Create one transaction row
        transaction = pd.DataFrame(
            [{
                "category": encoded_category,
                "amt": amount,
                "gender": encoded_gender,
                "state": encoded_state,
                "city_pop": city_population,
                "transaction_hour": transaction_hour,
                "transaction_day": transaction_day,
                "transaction_month": transaction_month,
                "customer_age": customer_age
            }]
        )

        # Ensure the feature order matches training
        transaction = transaction[
            feature_names
        ]

        # Scale numerical values
        numerical_columns = [
            "amt",
            "city_pop",
            "transaction_hour",
            "transaction_day",
            "transaction_month",
            "customer_age"
        ]

        transaction[
            numerical_columns
        ] = scaler.transform(
            transaction[
                numerical_columns
            ]
        )

        # Predict
        prediction = model.predict(
            transaction
        )[0]

        # Predict probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                transaction
            )[0]
        else:
            probability = None

        st.divider()

        # Display result
        if prediction == 1:

            st.error(
                "⚠️ Possible Fraudulent Transaction"
            )

            if probability is not None:
                st.metric(
                    "Fraud Probability",
                    f"{probability[1] * 100:.2f}%"
                )

            st.write(
                "The transaction contains patterns that the model "
                "associates with fraudulent activity."
            )

        else:

            st.success(
                "✅ Likely Legitimate Transaction"
            )

            if probability is not None:
                st.metric(
                    "Legitimate Probability",
                    f"{probability[0] * 100:.2f}%"
                )

            st.write(
                "The transaction does not appear to contain strong "
                "fraud indicators."
            )

    except ValueError as error:
        st.error(
            "The selected value does not match the values used "
            "during model training."
        )

        st.write(
            f"Technical details: {error}"
        )

    except Exception as error:
        st.error(
            f"Prediction error: {error}"
        )