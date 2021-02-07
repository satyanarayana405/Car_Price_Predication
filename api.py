import flask
from flask import request
from flask_cors import CORS
from flask import jsonify
from predict import predict_price

app = flask.Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

@app.route('/',methods=['GET'])
def home():
    return 'This is Car Selling price prediction app'

@app.route('/predict',methods=['GET'])
def api():
    dic={}
    dic['Car_Name'] = request.headers['Car_Name']
    dic['Year'] = request.headers['Year']
    dic['Present_Price'] = request.headers['Present_Price']
    dic['Kms_Driven'] = request.headers['Kms_Driven']
    dic['Fuel_Type'] = request.headers['Fuel_Type']
    dic['Seller_Type'] = request.headers['Seller_Type']
    dic['Transmission'] = request.headers['Transmission']
    dic['Owner'] = request.headers['Owner']
    prediction = predict_price(dic)
    return prediction

if __name__=='__main__':
    app.run(host='localhost',port=5000)


