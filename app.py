import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Library
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score

from sklearn.naive_bayes import GaussianNB

# import library yang di butuhkan
import requests
from bs4 import BeautifulSoup
import csv


# Stopwords 
import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
nltk.download('punkt')
# Download kamus stop words
nltk.download('stopwords')

# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Library Ekstaksi Fitur
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
# LDA
from sklearn.decomposition import LatentDirichletAllocation
import os

# Klasifikasi
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle


st.title("Website Applikasi Pengolahan dan Penambangan Web")
st.write("""
# Web Apps - Crowling Dataset
Applikasi Berbasis Web untuk melakukan **Sentiment Analysis**,
Jadi pada web applikasi ini akan bisa membantu anda untuk melakukan sentiment analysis mulai dari 
mengambil data (crowling dataset) dari sebuah website dan melakukan preprocessing dari data yang 
sudah diambil kemudian dapat melakukan klasifikasi serta dapat melihat akurasi dari model yang diinginkan.
### Menu yang disediakan dapat di lihat di bawah ini :
""")

# inisialisasi data 
# data = pd.read_csv("ecoli.csv")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Crowling Data", "LDA", "Modeling", "Evaluasi", "Implementasi"])

with tab1:

    st.subheader("Deskripsi")
    st.write("""Crowling Data adalah proses automatis untuk mengumpulkan dan mengindeks data dari 
    berbagai sumber seperti situs web, database, atau dokumen. Proses ini menggunakan software atau 
    aplikasi khusus yang disebut "crawler" untuk mengakses sumber data dan mengambil informasi yang 
    dibutuhkan. Data yang dikumpulkan melalui crawling kemudian dapat diproses dan digunakan untuk 
    berbagai tujuan, seperti analisis data, penelitian, atau pengembangan sistem informasi. Tujuanya
    untuk mengumpulkan data dari berbagai sumber dan mengindeksnya sehingga mudah untuk diakses 
    dan dianalisis.
    """)

    st.write("""
    ### Want to learn more?
    - Dataset (studi kasus) [kompas.com](https://www.kompas.com/)
    - Github Account [github.com](https://github.com/AliGhufron-28/datamaining)
    """)

    st.subheader("Crowling Data")
    fakultas = 4
    page = 1
    url = st.text_input("Masukkan Link website")

    hasil = st.button("submit")

    st.subheader("Data Hasil Crawling dengan 3 Kategori")
    data = pd.read_csv("data_berita_kompas.csv")
    st.write(data)
    data.shape

with tab2:
    st.subheader("Latent Dirichlet Allocation (LDA)")
    st.write("""Latent Dirichlet Allocation (LDA) adalah model probabilistik generatif dari koleksi 
    data diskrit seperti korpus teks. Ide dasarnya adalah bahwa dokumen direpresentasikan sebagai 
    campuran acak atas topik laten (tidak terlihat).""")

    st.write("""LDA merupakan model Bayesian hirarki tiga tingkat, di mana setiap item koleksi dimodelkan 
    sebagai campuran terbatas atas serangkaian set topik. Setiap topik dimodelkan sebagai campuran tak 
    terbatas melalui set yang mendasari probabilitas topik. Dalam konteks pembuatan model teks, 
    probabilitas topik memberikan representasi eksplisit dari sebuah dokumen.""")

    st.write("""data yang telah diambil kemudian dilakukan normalisasi untuk membuang atribute kata yang
    tidak diperlukan kemudian dilakukan ekstraksi fitur dengan tfidf kemudian baru implementasi LDA.""")

    st.subheader("Implementasi LDA dengan 3 Topik")
    st.write("parameter yang digunakan dalam LDA : K = 3, alpha = 0.1, beta = 0.2")
    data_topik = pd.read_csv("data_topik.csv")
    st.write(data_topik)
    data_topik.shape

    st.subheader("Data digabung dengan label")
    data_topik_label = pd.read_csv("data_3topik.csv")
    st.write(data_topik_label)
    data_topik_label.shape

with tab3:
    st.subheader("Modelling Menggunakan Naive Bayes Algorithm")
    st.write("""Algoritma Naive bayes merupakan metode pengklasifikasian berdasarkan probabilitas 
    sederhana dan dirancang agar dapat dipergunakan dengan asumsi antar variabel penjelas saling 
    bebas (independen). Pada algoritma ini pembelajaran lebih ditekankan pada pengestimasian 
    probabilitas. Keuntungan algoritma naive bayes adalah tingkat nilai error yang didapat lebih 
    rendah ketika dataset berjumlah besar, selain itu akurasi naive bayes dan kecepatannya lebih tinggi 
    pada saat diaplikasikan ke dalam dataset yang jumlahnya lebih besar.""")

    st.subheader("Hasil modeling dari data LDA")
    data_topik_label = pd.read_csv("data_3topik.csv")
    #Train and Test split
    X_topik = data_topik_label.drop('Category', axis=1)
    y_topik = data_topik_label['Category']
    X_topik_train,X_topik_test,y_topik_train,y_topik_test= train_test_split(X_topik,y_topik,test_size=0.1,random_state=42)
    
    # # ambil label
    model_nb = GaussianNB()
    filename = "model_naivebayes.pkl"

    model_nb.fit(X_topik_train,y_topik_train)
    Y_pred = model_nb.predict(X_topik_test)

    score=metrics.accuracy_score(y_topik_test,Y_pred)
    loaded_model = pickle.load(open(filename, 'rb'))
    st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)

with tab4:
    st.subheader("Evaluasi")

    data_topik_label = pd.read_csv("data_3topik.csv")
    st.write(data_topik_label)
    #Train and Test split
    X_topik = data_topik_label.drop('Category', axis=1)
    y_topik = data_topik_label['Category']
    X_topik_train,X_topik_test,y_topik_train,y_topik_test= train_test_split(X_topik,y_topik,test_size=0.1,random_state=42)
    
    # ambil label

    model_nb = GaussianNB()
    filename = "model_naivebayes.pkl"

    model_nb.fit(X_topik_train,y_topik_train)
    Y_pred = model_nb.predict(X_topik_test)

    score=metrics.accuracy_score(y_topik_test,Y_pred)
    loaded_model = pickle.load(open(filename, 'rb'))
    st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)

    report = classification_report(y_topik_test, Y_pred)
    st.write(f'Report Klasifikasi:\n{report}')

with tab5:
    st.subheader("Implementasi")

    st.subheader("Parameter Inputan")
    topik1 = st.number_input("Masukkan Nilai Topik 1 :")
    topik2 = st.number_input("Masukkan Nilai Topik 2 :")
    topik3 = st.number_input("Masukkan Nilai Topik 3 :")

    hasil = st.button("cek klasifikasi")

    if hasil:

        data_topik_label = pd.read_csv("data_3topik.csv")
        #Train and Test split
        X_topik = data_topik_label.drop('Category', axis=1)
        y_topik = data_topik_label['Category']
        X_topik_train,X_topik_test,y_topik_train,y_topik_test= train_test_split(X_topik,y_topik,test_size=0.1,random_state=42)
        
        # ambil label
        model_nb = GaussianNB()
        filename = "model_naivebayes.pkl"

        model_nb.fit(X_topik_train,y_topik_train)
        Y_pred = model_nb.predict(X_topik_test)

        score=metrics.accuracy_score(y_topik_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))

        dataArray = [topik1, topik2, topik3]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        st.write(f"Algoritma yang digunakan adalah = Naive Bayes")
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    
