import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import pickle
from datetime import date
import os
import warnings
warnings.filterwarnings('ignore')

def load_csv():
    file_path = os.path.abspath(os.path.dirname(__file__))+'\cardata.csv'
    data=pd.read_csv(file_path)
    return data

def preprocess(data,train=True):
    if train:
        data['no_of_years']=(date.today().year)-(data.Year)
    else:
        data['no_of_years'] = (date.today().year) - int(data.Year)
    data.drop('Year',axis=1,inplace=True)
    data['Fuel_Type']=data['Fuel_Type'].map({'Petrol':1,'Diesel':2,'CNG':3})
    data['Seller_Type']=data['Seller_Type'].map({'Individual':1,'Dealer':2})
    data['Transmission']=data['Transmission'].map({'Automatic':1,'Manual':2})
    return data

def Binary_enc(data):
    Bi_encoder = ce.BinaryEncoder(cols=['Car_Name'])
    Bi_encoder = Bi_encoder.fit(data)
    data = Bi_encoder.transform(data)
    with open('Bi_encoder.pickle', 'wb') as file:
        pickle.dump(Bi_encoder,file)
    return data

def split_data(df):
    x=df.loc[:,df.columns!='Selling_Price']
    y=df['Selling_Price']
    return x,y
    # y = list(y.values.ravel())

def minmax_scaling(x):
    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    x=minmax_scale.fit_transform(x)
    with open('minmax_scale.pickle', 'wb') as file:
        pickle.dump(minmax_scale,file)
    return x

def model_fit(x,y):
    regressor = RandomForestRegressor()
    model = regressor.fit(x,y)
    with open('model.pickle', 'wb') as file:
        pickle.dump(model,file)


def train(data=None,train=True):
    if train:
        data=load_csv()
        data=preprocess(data)
        x, y = split_data(data)
        x = Binary_enc(x)
        x=minmax_scaling(x)
        model_fit(x,y)
    else:
        data = preprocess(data,train=False)
        with open('Bi_encoder.pickle', 'rb') as file:
            Bi_encoder = pickle.load(file)
        data = Bi_encoder.transform(data)
        with open('minmax_scale.pickle', 'rb') as file:
            minmax_scale = pickle.load(file)
        data=minmax_scale.transform(data)
        return data

if __name__=='__main__':
    train()

