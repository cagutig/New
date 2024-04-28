# Importación librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
#import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse

best_model = joblib.load('model_deployment/car_price_reg.pkl')
label_encoders = joblib.load('model_deployment/label_encoders.pkl')

app = Flask(__name__)
api = Api(app, version='1.0', title='Model API',
          description='A simple API that use model to make predictions')

ns = api.namespace('predict', description='Model Prediction')

model = api.model('PredictionData', {
    'Year': fields.Integer(required=True, description='Year of the vehicle'),
    'Mileage': fields.Integer(required=True, description='Mileage of the vehicle'),
    'State': fields.String(required=True, description='State where the vehicle is registered'),
    'Make': fields.String(required=True, description='Make of the vehicle'),
    'Model': fields.String(required=True, description='Model of the vehicle'),
})

parser = reqparse.RequestParser()
parser.add_argument('Year', type=int, required=True, help='Year of the vehicle')
parser.add_argument('Mileage', type=int, required=True, help='Mileage of the vehicle')
parser.add_argument('State', type=str, required=True, help='State where the vehicle is registered')
parser.add_argument('Make', type=str, required=True, help='Make of the vehicle')
parser.add_argument('Model', type=str, required=True, help='Model of the vehicle')

@ns.route('/')
class CarPriceApi(Resource):
    @api.expect(model)
    @api.response(200, 'Success')
    def post(self):
        args = parser.parse_args()
        input_data = pd.DataFrame([args])

        # Aplicar label encoding
        for column in ['State', 'Make', 'Model']:
            input_data[column] = label_encoders[column].transform(input_data[column])

        # Predecir con el modelo
        prediction = best_model.predict(input_data)

        # Devolver el resultado
        return {
            "result": float(prediction[0])
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)