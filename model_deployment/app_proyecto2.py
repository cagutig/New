from flask import Flask, request, jsonify, render_template_string
from flask_restx import Api, Resource, fields
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

# Inicializar Flask y Flask-RESTX
app = Flask(__name__)
api = Api(app, version='1.0', title='Movie Genre Classification API',
          description='API para clasificar géneros de películas basado en el título y la sinopsis.', doc='/docs')

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
        
        # Ordenar las predicciones en orden descendente
        sorted_prediction = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))
        
        return jsonify({'prediction': sorted_prediction})

@app.route('/')
def home():
    form_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Genre Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }
            .container {
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            label {
                display: block;
                margin: 15px 0 5px;
            }
            input[type="text"],
            textarea {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                display: block;
                width: 100%;
                padding: 10px;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
            }
            .result {
                margin-top: 20px;
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Movie Genre Prediction</h1>
            <form action="/predict" method="post">
                <label for="title">Title:</label>
                <input type="text" id="title" name="title" required>

                <label for="plot">Plot:</label>
                <textarea id="plot" name="plot" rows="4" required></textarea>

                <button type="submit">Predict</button>
            </form>
            {% if prediction %}
            <div class="result">
                <h2>Prediction Result:</h2>
                <pre>{{ prediction }}</pre>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict_form():
    title = request.form['title']
    plot = request.form['plot']
    title_plot = f"{title} {plot}"
    
    # Vectorizar el texto usando el vectorizador TF-IDF
    input_data = vectorizer.transform([title_plot]).toarray()
    
    # Hacer la predicción
    prediction = model.predict(input_data)[0]
    
    # Asociar las predicciones con los géneros
    prediction_dict = {genre: round(float(pred), 4) for genre, pred in zip(genres, prediction)}
    
    # Ordenar las predicciones en orden descendente
    sorted_prediction = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))
    
    form_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Genre Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }
            .container {
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            label {
                display: block;
                margin: 15px 0 5px;
            }
            input[type="text"],
            textarea {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                display: block;
                width: 100%;
                padding: 10px;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
            }
            .result {
                margin-top: 20px;
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Movie Genre Prediction</h1>
            <form action="/predict" method="post">
                <label for="title">Title:</label>
                <input type="text" id="title" name="title" required>

                <label for="plot">Plot:</label>
                <textarea id="plot" name="plot" rows="4" required></textarea>

                <button type="submit">Predict</button>
            </form>
            {% if prediction %}
            <div class="result">
                <h2>Prediction Result:</h2>
                <pre>{{ prediction }}</pre>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(form_html, prediction=sorted_prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
