import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle 
import tensorflow as tf
import joblib

def run_ml_app():
    st.subheader('Mechine reaning')

    #1.유저한테 입력을 받는다.
    #성별입력
    gender = st.radio('성별을 선택하세요',['남자','여자'])
    if gender == '남자':
        gender = 1
    else:
        gender = 0
    
    age = st.number_input('나이 입력',min_value=0,max_value=120)

    salary = st.number_input('연봉 입력',min_value=0)

    debt = st.number_input('빚 입력',min_value=0)

    worth = st.number_input('자산 입력',min_value=0)

    #예측한다.
    #2-1 모델 불러오기
    model = tf.keras.models.load_model('data/car_ai.h5')
    
    #2-2 넘파이 어레이 만든다.
    new_data=np.array([gender,age,salary,debt,worth]).reshape(1,-1)
    #2-3 X데이터 피처스케일링
    sc_x = joblib.load('data/sc_X.pk1')

    new_data = sc_x.transform(new_data)
    #2-4 예측한다.
    y_pred = model.predict(new_data)
    #예측결과는 스케일링 된 결과이므로 다시 돌려야한다.
    #st.write(y_pred[0][0])

    sc_y = joblib.load('data/sc_y.pk1')

    y_pred_original = sc_y.inverse_transform(y_pred)
    btn = st.button('결과 보기')
    if btn :
        st.write('예측 결과입니다. {:,.1f}달러의 차를 사실 수 있습니다.'.format(y_pred_original[0][0]))
    #st.write(y_pred_original)



    



