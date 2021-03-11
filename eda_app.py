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
def run_eda_app():
    st.subheader('EDA 화면입니다.')

    car_df=pd.read_csv('data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')
    
    radio_menu = ['데이터프레임','통계치']
    
    selected_radio = st.radio('선택하세요',radio_menu)
    if selected_radio == '데이터프레임':

        st.dataframe(car_df)
    elif selected_radio == '통계치':
        st.dataframe(car_df.describe())
    
    columns = car_df.columns
    columns = list(columns)
    selected_columns = st.multiselect('컬럼명을 선택해주세요',columns)
    if len(selected_columns) != 0:
        st.dataframe(car_df[selected_columns])
    else :
        st.write('선택한 컬럼이 없습니다.')

    # 상관계수를 화면에 보여주도록 만듭니다.
    # 멀티셀렉트에 컬럼명을 보여주고,해당컬럼들에 대한 상관계수를 보여주세요
    # 단,컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야합니다.

    #columns_list = car_df.loc[]
    print(car_df.dtypes != object)
    corr_columns = car_df.columns[car_df.dtypes != object]
    selected_columns_corr = st.multiselect('상관계수 컬럼 선택해주세요',corr_columns)
    if len(selected_columns_corr) != 0:
        st.dataframe(car_df[selected_columns_corr].corr())
        
        fig=sns.pairplot(data=car_df[selected_columns_corr])
        st.pyplot(fig)        
        
    else :
        st.write('선택한 컬럼이 없습니다.')
    
    #컬럼을 하나만 선택하면, 해당 컬럼의 min과 max에 해당하는 사람의 데이터를 화면에 보여준다.
    
   
    min_max_columns = car_df.columns[car_df.dtypes == float]
    selected_columns_min_max = st.selectbox('컬럼을 선택해주세요',min_max_columns)
    st.write('max값',car_df[selected_columns_min_max].max())
    st.write('min값',car_df[selected_columns_min_max].min())
    

    #고객이름을 검색할 수 있는 기능 개발

    # text_serch=st.text_input('이름을 입력하세요')
    # # name=car_df['Customer Name']
    # # name.apply(str.lower)
    # # print(car_df['Customer Name'].unique())
    # # car_df['Customer Name'].unique()
    # name_write=st.write(text_serch)
    # if name_write == car_df['Customer Name'].unique():
    #     st.write()

    word = st.text_input('검색어를 입력하세요')

    result=car_df.loc[car_df['Customer Name'].str.contains(word,case=False),]
    
    st.dataframe(result)


    