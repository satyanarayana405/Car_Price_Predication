import pickle
import sys
import pandas as pd
from train import train

def predict_price(dic):
    with open('model.pickle','rb') as file:
        model=pickle.load(file)
    df=pd.DataFrame(dic,index=[0])
    data=train(data=df,train=False)
    prediction=model.predict(data)
    return str(round(prediction[0],2))

