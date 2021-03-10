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


def main():
    df=pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')
    # lang = [']
    # # selected_lang = st.selectbox("언어 선택하세요",lang)
    # # st.write('당신이 선택한 는 {}입니다.'.format(selected_lang))
    
    
    st.dataframe(df)

    #X,y값 나누어주기
    X = df.iloc[:,3:-2+1]
    st.write(X)
    y = df['Car Purchase Amount']
    st.write(y)
    
    #X값 피처스케일링
    sc_X=MinMaxScaler()
    X=sc_X.fit_transform(X)
    st.write(X)

    #y값 차원변경
    y=np.array([y])
    y=y.reshape(-1,1)
    st.write(y)

    #y값 피처스케일링
    sc_y=MinMaxScaler()
    y=sc_y.fit_transform(y)
    st.write(y)

    #트레이닝셋과 테스트셋 분리하기
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

    # #모델링 하기
    # model = Sequential()
    # model.add(Dense(25,input_dim=5,activation ='relu'))
    # model.add(Dense(40,activation = 'relu'))
    # model.add(Dense(1,activation = 'linear'))

    # model.summary()
    # #가장 좋은 모델과 로그 저장
    
    # # CHECKPOINT_PATH ='C:/Users/5-13/Documents/Streamlit/day04/data/block.h5'
    # # LOGFILE_PATH ='C:/Users/5-13/Documents/Streamlit/day04/data1/logblock.csv'
    # # # CHECKPOINT_PATH ='day04/data/block.h5'
    # # # LOGFILE_PATH ='day04/data1/logblock.csv'

    # # cp = ModelCheckpoint(filepath=CHECKPOINT_PATH,monitor='loss',save_best_only=True,verbose=1)
    # # csv_logger = CSVLogger(filename=LOGFILE_PATH,append=True)

    # #컴파일러하기
    # model.compile(optimizer='adam',loss='mean_squared_error')

    # #학습시키기
    # history = model.fit(X_train,y_train,epochs=9,batch_size=10,verbose=1)

    #y_pred=model.predict(X_test)

    # ret_df=pd.DataFrame({'실제값':y_test.reshape(-1,),'예측값': y_pred.reshape(-1,)})
    # st.write(ret_df)   

       
    CHECKPOINT_PATH = 'C:/Users/5-13/Documents/Streamlit/day04/checkpoints/by-type-best_model-block-1-2.h5'
    model=tf.keras.models.load_model(CHECKPOINT_PATH)
    
    y_pred=model.predict(X_test)
    
    
    ret_df=pd.DataFrame({'실제값':y_test.reshape(-1,),'예측값': y_pred.reshape(-1,)})
    st.write(ret_df)   


    lang = ['데이터프레임','X값','y',]
    selected_lang = st.selectbox("데이터 프레임 선택하세요",lang)
    if selected_lang == '데이터프레임':
        st.dataframe(df)
    
    elif selected_lang == 'X값':
        sc_X.inverse_transform(X)
    
    

    

    

    
if __name__ == '__main__' :
    main()  

