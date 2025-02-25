import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import re
from io import BytesIO

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Utility functions
def log_action(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append(f"{timestamp} - {action}")

# Main app
st.title("📊 Data Sweeper")
st.subheader("Clean, Transform & Optimize Your Datasets")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    upload_option = st.radio("Data Source", ["Upload File", "Sample Data"])
    
    if upload_option == "Sample Data":
        sample_dataset = st.selectbox("Choose Sample", ["Titanic", "Iris", "Tips"])
    
    st.divider()
    st.write("Developed by Kumail Ali")

# Data Loading Section
st.header("Step 1: Data Ingestion")
if upload_option == "Upload File":
    uploaded_file = st.file_uploader("Upload Dataset", 
                                   type=["csv", "xlsx", "json"],
                                   accept_multiple_files=False)
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state.df = df.copy()
            log_action(f"Uploaded file: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
else:
    if sample_dataset == "Titanic":
        df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
    elif sample_dataset == "Iris":
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                         names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    elif sample_dataset == "Tips":
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    
    st.session_state.df = df.copy()
    log_action(f"Loaded sample dataset: {sample_dataset}")

# Show raw data preview
if st.session_state.df is not None:
    st.subheader("Data Preview")
    st.write(f"Shape: {st.session_state.df.shape}")
    st.dataframe(st.session_state.df.head())

    # Data Cleaning Section
    st.header("Step 2: Data Cleaning")
    
    cleaning_options = st.multiselect("Select Cleaning Operations",
                                     ["Remove Duplicates", 
                                      "Handle Missing Values",
                                      "Standardize Data Types",
                                      "Clean Text Data"])
    
    # Duplicate Removal
    if "Remove Duplicates" in cleaning_options:
        duplicates = st.session_state.df.duplicated().sum()
        if duplicates > 0:
            if st.checkbox(f"Remove {duplicates} duplicates?"):
                st.session_state.df = st.session_state.df.drop_duplicates()
                log_action(f"Removed {duplicates} duplicate rows")
    
    # Missing Values Handling
    if "Handle Missing Values" in cleaning_options:
        st.subheader("Missing Values Treatment")
        missing = st.session_state.df.isna().sum()
        st.write("Missing values per column:")
        st.write(missing[missing > 0])
        
        for col in missing[missing > 0].index:
            treatment = st.selectbox(
                f"Treatment for {col}",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
                key=f"missing_{col}"
            )
            
            if treatment == "Drop rows":
                st.session_state.df = st.session_state.df.dropna(subset=[col])
                log_action(f"Dropped rows with missing {col}")
            else:
                if treatment == "Fill with mean":
                    fill_value = st.session_state.df[col].mean()
                elif treatment == "Fill with median":
                    fill_value = st.session_state.df[col].median()
                else:
                    fill_value = st.session_state.df[col].mode()[0]
                
                st.session_state.df[col] = st.session_state.df[col].fillna(fill_value)
                log_action(f"Filled missing {col} with {fill_value}")
    
    # Data Type Standardization
    if "Standardize Data Types" in cleaning_options:
        st.subheader("Data Type Conversion")
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        date_cols = st.session_state.df.select_dtypes(include='datetime').columns.tolist()
        text_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()
        
        for col in st.session_state.df.columns:
            current_type = st.session_state.df[col].dtype
            new_type = st.selectbox(
                f"{col} ({current_type})",
                ["Keep as is", "Numeric", "Datetime", "Category", "String"],
                key=f"dtype_{col}"
            )
            
            if new_type != "Keep as is":
                try:
                    if new_type == "Numeric":
                        st.session_state.df[col] = pd.to_numeric(st.session_state.df[col])
                    elif new_type == "Datetime":
                        st.session_state.df[col] = pd.to_datetime(st.session_state.df[col])
                    elif new_type == "Category":
                        st.session_state.df[col] = st.session_state.df[col].astype('category')
                    elif new_type == "String":
                        st.session_state.df[col] = st.session_state.df[col].astype('string')
                    
                    log_action(f"Converted {col} from {current_type} to {new_type}")
                except Exception as e:
                    st.error(f"Error converting {col}: {str(e)}")
    
    # Text Cleaning
    if "Clean Text Data" in cleaning_options:
        st.subheader("Text Data Cleaning")
        text_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()
        
        for col in text_cols:
            st.write(f"Cleaning {col}")
            if st.checkbox(f"Trim whitespace in {col}", key=f"trim_{col}"):
                st.session_state.df[col] = st.session_state.df[col].str.strip()
                log_action(f"Trimmed whitespace in {col}")
            
            if st.checkbox(f"Standardize case in {col}", key=f"case_{col}"):
                st.session_state.df[col] = st.session_state.df[col].str.title()
                log_action(f"Standardized case in {col}")

# Data Validation Section
st.header("Step 3: Data Validation")
if st.session_state.df is not None:
    validation_col = st.selectbox("Select column to validate", st.session_state.df.columns)
    
    if st.session_state.df[validation_col].dtype == 'object':
        if st.checkbox("Validate Email Format"):
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            invalid_emails = ~st.session_state.df[validation_col].str.match(email_pattern, na=False)
            st.write(f"Found {invalid_emails.sum()} invalid email addresses")
    
    if np.issubdtype(st.session_state.df[validation_col].dtype, np.number):
        min_val = st.number_input("Minimum allowed value", value=0)
        max_val = st.number_input("Maximum allowed value", value=100)
        invalid_values = ~st.session_state.df[validation_col].between(min_val, max_val)
        st.write(f"Found {invalid_values.sum()} values outside range {min_val}-{max_val}")

# Data Export Section
st.header("Step 4: Export Cleaned Data")
if st.session_state.df is not None:
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    if export_format == "CSV":
        output = st.session_state.df.to_csv(index=False).encode()
    elif export_format == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.df.to_excel(writer, index=False)
        output = output.getvalue()
    else:
        output = st.session_state.df.to_json(indent=2).encode()
    
    st.download_button(
        label=f"Download {export_format}",
        data=output,
        file_name=f"cleaned_data.{export_format.lower()}",
        mime="application/octet-stream"
    )

# Show logs
st.sidebar.header("Processing Logs")
for log in reversed(st.session_state.logs[-10:]):  # Show last 10 logs
    st.sidebar.code(log, language="text")