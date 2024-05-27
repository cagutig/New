from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
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
model_input = api.model('PredictionData', {
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

# Lista de géneros
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
          'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
          'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

# Definir la clase para la predicción
@ns.route('/')
class MovieGenreApi(Resource):
    @api.expect(model_input)
    @api.response(200, 'Success')
    def post(self):
        data = request.get_json(force=True)
        title = data['title']
        plot = data['plot']
        title_plot = f"{title} {plot}"
        
        # Vectorizar el texto usando el vectorizador TF-IDF
        input_data = vectorizer.transform([title_plot]).toarray()
        
        # Hacer la predicción
        prediction = model.predict(input_data)[0]
        
        # Asociar las predicciones con los géneros
        prediction_dict = {genre: round(float(pred), 4) for genre, pred in zip(genres, prediction)}
        
        # Ordenar las predicciones en orden descendente por valores numéricos
        sorted_prediction = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))
        
        return jsonify({'prediction': sorted_prediction})

# Ruta para verificar el funcionamiento de la API
@app.route('/')
def home():
    return "La API está funcionando correctamente. Visita /docs para acceder a la interfaz Swagger UI."

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
