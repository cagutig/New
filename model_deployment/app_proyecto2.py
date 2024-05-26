# Importar librerías
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, reqparse
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

# Inicializar Flask y Flask-RESTX
app = Flask(__name__)
api = Api(app, version='1.0', title='Movie Genre Classification API',
          description='API para clasificar géneros de películas basado en el título y la sinopsis.')

# Definir el namespace
ns = api.namespace('predict', description='Predicción del modelo')

# Definir el modelo de datos esperado
model = api.model('PredictionData', {
    'title': fields.String(required=True, description='Título de la película'),
    'plot': fields.String(required=True, description='Sinopsis de la película'),
})

# Cargar el modelo de red neuronal entrenado y recompilarlo
model_path = 'corrected_model.h5'
model = load_model(model_path)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Cargar el vectorizador TF-IDF
vectorizer_path = 'tfidf_vectorizer.pkl'
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Definir el parser para los argumentos
parser = reqparse.RequestParser()
parser.add_argument('title', type=str, required=True, help='Título de la película')
parser.add_argument('plot', type=str, required=True, help='Sinopsis de la película')

# Definir la clase para la predicción
@ns.route('/')
class MovieGenreApi(Resource):
    @api.expect(model)
    @api.response(200, 'Success')
    def post(self):
        args = parser.parse_args()
        title = args['title']
        plot = args['plot']
        title_plot = f"{title} {plot}"
        
        # Vectorizar el texto usando el vectorizador TF-IDF
        input_data = vectorizer.transform([title_plot]).toarray()
        
        # Hacer la predicción
        prediction = model.predict(input_data)
        
        # Convertir la predicción a una lista para que sea serializable en JSON
        prediction = prediction.tolist()
        
        return jsonify({'prediction': prediction})

# Ruta para verificar el funcionamiento de la API
@app.route('/')
def home():
    return "La API está funcionando correctamente."

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

