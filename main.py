import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

st.write("""
    # Program Prediksi Peluang Penerimaan Program S1
""")

st.write( """
    ## Keterangan Data Yang Digunakan
    1. Average raport Merupakan rata-rata nilai raport semester 1-5 Untuk Masuk Program S1 (0 - 340) Bersifat Continous
    2. T0 Score Merupakan Score Kemampuan TKA dan TPS(0 - 120) Bersifat Continous
    3. Rating Sekolah (0 - 5) Bersifat Ordinal
    4. Kekuatan Surat Rekomendasi sekolah (0 - 5) Bersifat Ordinal
    5. Pengalaman Organisasi (0 -1)
    6. Nilai UN merupakan nilai yang didapatkan setelah selesai menempuh ujian nasional
   
    6. Peluang Diterima (0 - 1) Merupakan Dependent Variable

""")

st.write("""
    ## Overview Data
""")

myData = pd.read_csv(r"D:\kelulusan\data.csv")

st.dataframe(myData)

st.write("""
    ## Deskripsi Data
""")

st.dataframe(myData.describe())

# Preproccessing Data

st.write("""
    ## Dilakukan Preprocessing Data dimana Fitur dan Labelnya akan Dipisah
""")

# Memisahkan Label Dan Fitur 
X = myData.iloc[:, 1:-1].values
y = myData.iloc[:, -1].values



st.write("## Input Data X",X)
st.write("## Label Data y",y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)



from sklearn.preprocessing import StandardScaler 

ss_train_test = StandardScaler()


X_train_ss_scaled = ss_train_test.fit_transform(X_train)
X_test_ss_scaled = ss_train_test.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

l_regressor_ss = LinearRegression()
l_regressor_ss.fit(X_train_ss_scaled, y_train)
y_pred_l_reg_ss = l_regressor_ss.predict(X_test_ss_scaled)

st.write("Dengan Menggunakan Multiple Linear Regression Diperoleh Skor Untuk Data Test")
st.write(r2_score(y_test, y_pred_l_reg_ss))


st.write("# Sekarang Silahkan Masukan Skor Test Kamu Untuk Mengetahui Prediksi Peluang Kelulusan S1 Kamu")


form = st.form(key='my-form')
inputAvg = form.number_input("Masukan Average Raport School: ", 0)
inputTO = form.number_input("Masukan TO Score: ", 0)
inputschoolRating = form.number_input("Masukan Rating School: ", 0)
inputP = form.number_input("Masukan Kekuatan Portofolio: ", 0)
inputRecom = form.number_input("Masukan Kekuatan Recomendation School: ", 0)
inputAvgUN = form.number_input("Masukan nilai Avg UN: ", 0)
inputOrganization = form.number_input("Pengalaman Organisasi, 1 Jika Pernah Organisasi, 0 Jika Tidak", 0)
submit = form.form_submit_button('Submit')

completeData = np.array([inputAvg, inputTO, inputschoolRating, 
                        inputP, inputRecom, inputAvgUN, inputOrganization]).reshape(1, -1)
scaledData = ss_train_test.transform(completeData)


st.write('Tekan Submit Untuk Melihat Prediksi Peluang S1 Anda')

if submit:
    prediction = l_regressor_ss.predict(scaledData)
    if prediction > 1 :
        result = 1
    elif prediction < 0 :
        result = 0
    else :
        result = prediction[0]
    st.write(result*100, "Percent")
