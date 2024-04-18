import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import *
import pickle
import altair as alt

    # Fungsi untuk memuat data
@st.cache_data
def load_data_raw():
    df = pd.read_csv("Telco_customer_churn.csv")
    return df

def load_data_clean():
    df = pd.read_csv("Data Cleaned (1).csv")
    return df

def main():
    # Judul halaman
    st.title("Telco Customer Churn Analysis")

    # Pilihan menu di sidebar
    with st.sidebar :
        selected = option_menu('Doni',['Beranda','Distribusi','Perbandingan','Komposisi','Relasi','Prediksi'],default_index=0)

    # Menampilkan konten sesuai pilihan menu
    if selected == "Beranda":
        st.header("Selamat datang di aplikasi analisis telco customer churn!")
        st.write("Di sini Anda dapat menganalisis berbagai aspek yang mempengaruhi kemungkinan churn.")

    elif selected == "Distribusi":
        st.header("Analisis Distribusi")
        st.write("Di sini Anda dapat melihat distribusi dari Churn Label yang ada dalam dataset.")
        distribusi(load_data_raw())
        
    elif selected == "Perbandingan":
        st.header("Analisis Perbandingan")
        st.write("Di sini Anda dapat melakukan perbandingan antara berbagai variabel dalam dataset.")
        perbandingan(load_data_raw())
   
    elif selected == "Komposisi":
        st.header("Analisis Komposisi")
        st.write("Di sini Anda dapat melakukan perbandingan antara berbagai variabel dalam dataset.")
        komposisi(load_data_raw())
       

    elif selected == "Relasi":
        st.header("Analisis Relasi")
        st.write("Di sini Anda dapat melihat komposisi dari berbagai kategori dalam dataset.")
        relasi(load_data_raw())
       

    elif selected == "Prediksi":
        st.header("Prediksi")
        st.write("Di sini Anda dapat melakukan prediksi berdasarkan data yang ada.")
        prediksi()

def distribusi(data):
    Churn_Label_counts = data['Churn Label'].value_counts()

    # Menampilkan diagram pie menggunakan Streamlit
    st.title('Persebaran Data Churn Label')
    st.write("Jumlah 'Yes':", Churn_Label_counts['Yes'])
    st.write("Jumlah 'No':", Churn_Label_counts['No'])

    # Membuat plot pie dengan Altair
    chart_data = pd.DataFrame({'labels': Churn_Label_counts.index, 'values': Churn_Label_counts.values})
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='labels',
        y='values',
        color='labels'
    ).properties(
        width=400,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def perbandingan(data):
    Churn_Reason_counts = data['Churn Reason'].value_counts(ascending=False)

    # Menampilkan diagram bar menggunakan Streamlit
    st.title('Persebaran Churn Reason')
    st.bar_chart(Churn_Reason_counts)

    # Membuat plot bar dengan Altair
    chart_data = pd.DataFrame({'Churn Reason': Churn_Reason_counts.index, 'Jumlah': Churn_Reason_counts.values})
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='Jumlah',
        y=alt.Y('Churn Reason', sort='-x'),
        color='Jumlah:Q',
        text='Jumlah'
    ).properties(
        width=600,
        height=600
    )
    st.altair_chart(chart, use_container_width=True)

def komposisi(data):
    contingency_table = pd.crosstab(data['Churn Reason'], data['City'])
    top_10_cities = contingency_table.sum().nlargest(10).index
    contingency_table_top_10 = contingency_table[top_10_cities]
    contingency_table_top_10 = contingency_table_top_10.reset_index()

    st.title('Churn Reason dari 10 Kota')
    st.write('Visualisasi data Churn Reason dari 10 kota teratas')

    fig = px.bar(contingency_table_top_10, x='Churn Reason', y=top_10_cities,
                title='Churn Reason dari 10 Kota',
                labels={'Churn Reason': 'Churn Reason', 'value': 'Jumlah', 'variable': 'Kota'},
                barmode='stack')
    fig.update_layout(width=900, height=600)
    st.plotly_chart(fig)

def relasi(data):
    numeric_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()

    # Menampilkan judul
    st.title('Relasi Antar Kolom')

    # Memilih kolom yang akan ditampilkan dalam heatmap
    selected_columns = st.multiselect('Pilih kolom', numeric_columns)

    # Membuat heatmap relasi antar kolom
    if selected_columns:
        selected_data = data[selected_columns]
        corr = selected_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt)
    else:
        st.write('Tidak ada kolom numerik yang dipilih. Silakan pilih setidaknya satu kolom numerik untuk ditampilkan dalam heatmap.')

def prediksi():
    gender = st.selectbox('Select gender',['Male','Female'])
    partner = st.selectbox('Select partner', ('Yes', 'No'))
    dependents = st.selectbox('Select dependents', ('Yes', 'No'))
    tenure = st.slider('Select tenure months',0,72)
    phone = st.selectbox('Select phone service', ('Yes', 'No'))
    multiple = st.selectbox('Select multiple lines', ('Yes', 'No', 'No Phone Services'))
    internet = st.selectbox('Select internet service', ('DSL', 'Fiber optic', 'No'))
    security = st.selectbox('Select online security', ('Yes', 'No', 'No Internet Services'))
    backup = st.selectbox('Select online backup', ('Yes', 'No', 'No Internet Services'))
    device = st.selectbox('Select device protection', ('Yes', 'No', 'No Internet Services'))
    tech = st.selectbox('Select tech support', ('Yes', 'No', 'No Internet Services'))
    streamingTv = st.selectbox('Select streaming TV', ('Yes', 'No', 'No Internet Services'))
    streamingMv = st.selectbox('Select streaming Movies', ('Yes', 'No', 'No Internet Services'))
    contract = st.selectbox('Select contract', ('Month-to-month', 'Two year', 'One year'))
    paper = st.selectbox('Select paperless billing', ('Yes', 'No'))
    charges = st.number_input('Input monthly charges')

    data = pd.DataFrame({
        'Gender' : [0 if gender == 'Male' else 1],
        'Partner' : [0 if partner == 'No' else 1],
        'Dependents' : [0 if dependents == 'No' else 1],
        'Tenure Months' : [tenure],
        'Phone Service' : [0 if phone == 'No' else 1],
        'Multiple Lines' : [1 if multiple == 'Yes' else 0],
        'Internet Service' : [0 if internet == 'DSL' else (1 if internet == 'Fiber Optic' else 2)],
        'Online Security' : [1 if security == 'Yes' else 0],
        'Online Backup' : [1 if backup == 'Yes' else 0],
        'Device Protection' : [1 if device == 'Yes' else 0],
        'Tech Support' : [1 if tech == 'Yes' else 0],
        'Streaming TV' : [1 if streamingTv == 'Yes' else 0],
        'Streaming Movies' : [1 if streamingMv == 'Yes' else 0],
        'Contract' : [0 if contract == 'Month-to-month' else (1 if contract == 'Two year' else 2)],
        'Paperless Billing' : [0 if paper == 'No' else 1],
        'Monthly Charges' : [0 if charges == 'No' else 1]
    })
    st.write(data)
    button = st.button('Predict')
    if (button):
        with open('gnb.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        predicted = loaded_model.predict(data)
        if (predicted[0] == 0):
            st.success(predicted)
        elif (predicted[0] == 1):
            st.success(predicted)
        else :
            st.error('Not Defined')

if __name__ == '__main__':
    main()