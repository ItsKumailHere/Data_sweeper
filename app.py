import streamlit as st
import pandas as pd
from io import BytesIO
import base64

def convert_csv_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def convert_excel_to_csv(file):
    df = pd.read_excel(file)
    return df.to_csv(index=False).encode()

def clean_data(df):
    # Basic data cleaning
    df = df.dropna()  # Remove rows with NaN values
    df = df.drop_duplicates()  # Remove duplicate rows
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()  # Strip whitespace
    return df

def visualize_data(df):
    st.subheader('Data Visualization')
    if not df.empty:
        st.line_chart(df.select_dtypes(include=['float64', 'int64']))
        st.bar_chart(df.select_dtypes(include=['float64', 'int64']))

def main():
    st.title('Data Sweeper Web Application')
    
    # File uploader for CSV or Excel
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            st.write("CSV file uploaded")
        else:
            df = pd.read_excel(uploaded_file)
            st.write("Excel file uploaded")
        
        # Clean data
        cleaned_df = clean_data(df)
        st.write("Cleaned Data Preview:")
        st.write(cleaned_df.head())
        
        # Data Visualization
        visualize_data(cleaned_df)
        
        # Convert file
        if st.button('Convert to CSV'):
            csv = convert_excel_to_csv(uploaded_file) if file_extension == 'xlsx' else cleaned_df.to_csv(index=False).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        if st.button('Convert to Excel'):
            excel = convert_csv_to_excel(cleaned_df) if file_extension == 'csv' else cleaned_df.to_excel(BytesIO(), index=False).getvalue()
            b64 = base64.b64encode(excel).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="output.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()