import streamlit as st
import pandas as pd
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('C:/Users/HP/Desktop/updated_model.keras')

# Initialize session state for transaction history if not already done
if 'transaction_history' not in st.session_state:
    st.session_state['transaction_history'] = []

# Define prediction function
def make_prediction(input_data):
    if len(input_data) < 12:
        input_data.extend([0] * (12 - len(input_data)))  # Add zeros for missing features
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0][0]  # Adjust indexing if model output shape differs

# Streamlit App with white and blue background
st.set_page_config(page_title="Fraud Detection System")
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom, #ffffff, #e0f7fa);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("Fraud Detection System")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Us", "Prediction", "Transaction History", "Bulk Upload"])

# Home Page
if page == "Home":
    st.header("Welcome to the Fraud Detection System")
    st.write("""
        This application helps detect potential fraudulent transactions using a machine learning model. 
        It provides real-time predictions for single transactions and bulk processing for multiple transactions. 
        Key features include:
        
        - **Single Transaction Prediction**: Use the 'Prediction' page to check if a transaction is potentially fraudulent.
        - **Bulk Upload**: Analyze multiple transactions at once by uploading a CSV file on the 'Bulk Upload' page.
        - **Transaction History**: View past predictions and transaction history.
        
        Navigate to the appropriate page using the sidebar to start using the app.
    """)

# About Us Page
elif page == "About Us":
    st.header("About the Fraud Detection System")
    st.write("""
        The Fraud Detection System was developed to help organizations and individuals identify potentially fraudulent transactions.
        
        - **Machine Learning Approach**: This app uses a neural network model trained on transaction data to predict fraud likelihood based on various factors like amount, balance, and transaction history.
        - **Goal**: To provide a tool that helps users detect unusual patterns in financial data and take preventive actions.
        
        **Disclaimer**: This tool provides predictions based on historical patterns and is not a substitute for professional fraud analysis. Always verify suspicious transactions with additional sources.
    """)

# Prediction Page
elif page == "Prediction":
    st.header("Make a Prediction")

    # Input fields for Prediction
    amount = st.number_input("Enter Amount:", min_value=0.0)
    old_balance = st.number_input("Enter Old Balance (Original):", min_value=0.0)
    new_balance = st.number_input("Enter New Balance (Original):", min_value=0.0)
    
    # Transaction Type dropdown
    transaction_type = st.selectbox("Select Transaction Type:", ["Credit", "Debit", "Transfer", "Withdrawal"])

    # Collect the input data in a list and fill in placeholders for remaining fields
    input_data = [amount, old_balance, new_balance]

    # Prediction button
    if st.button("Predict"):
        try:
            result = make_prediction(input_data)
            prediction_text = "Fraud" if result > 0.5 else "No Fraud"

            if result > 0.5:
                st.error("Alert: High likelihood of fraud!")
            else:
                st.success("Prediction indicates no fraud.")

            # Save the prediction to transaction history
            st.session_state['transaction_history'].append({
                'Amount': amount,
                'Old Balance': old_balance,
                'New Balance': new_balance,
                'Transaction Type': transaction_type,
                'Prediction': prediction_text
            })

            # Option to save and download
            if st.checkbox("Save results"):
                input_data_dict = {
                    'amount': amount,
                    'old_balance': old_balance,
                    'new_balance': new_balance,
                    'transaction_type': transaction_type,
                    'prediction': prediction_text
                }
                df = pd.DataFrame([input_data_dict])
                df.to_csv("prediction_results.csv", index=False)
                st.write("Saved as 'prediction_results.csv'")

                # Download button
                st.download_button(
                    label="Download Prediction Results",
                    data=df.to_csv(index=False),
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Transaction History Page
elif page == "Transaction History":
    st.header("Transaction History")
    st.write("View your previous transaction predictions here.")
    
    # Display transaction history from session state
    if st.session_state['transaction_history']:
        history_df = pd.DataFrame(st.session_state['transaction_history'])
        st.dataframe(history_df)

        # Option to download transaction history as CSV
        st.download_button(
            label="Download Transaction History",
            data=history_df.to_csv(index=False),
            file_name="transaction_history.csv",
            mime="text/csv"
        )
    else:
        st.write("No transactions recorded yet.")

# Bulk Upload Page
elif page == "Bulk Upload":
    st.header("Bulk Upload for Fraud Detection")
    st.write("Upload a CSV file with transaction data to get batch predictions.")

    uploaded_file = st.file_uploader("Choose a file", type="csv")
    
    if uploaded_file:
        # Read uploaded file as DataFrame
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())  # Display preview of the uploaded data
        
        # Ensure the uploaded file has the required 12 features
        if data.shape[1] != 12:
            st.error("The uploaded file does not have the required 12 columns.")
        else:
            # Make predictions on the bulk data
            predictions = model.predict(data)
            data['Fraud Prediction'] = ["Fraud" if pred > 0.5 else "No Fraud" for pred in predictions.flatten()]

            st.write("Predictions:")
            st.dataframe(data)

            # Download button for the bulk predictions
            st.download_button(
                label="Download Bulk Prediction Results",
                data=data.to_csv(index=False),
                file_name="bulk_prediction_results.csv",
                mime="text/csv"
            )







