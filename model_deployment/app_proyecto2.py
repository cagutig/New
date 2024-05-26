from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

app = Flask(__name__)

# Cargar el modelo de red neuronal entrenado y recompilarlo
model_path = 'corrected_model.h5'
model = load_model(model_path)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Cargar el vectorizador TF-IDF
vectorizer_path = 'tfidf_vectorizer.pkl'
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Verificar que 'title' y 'plot' estén en los datos
        if 'title' not in data or 'plot' not in data:
            return jsonify({'error': 'Title and plot are required'}), 400

        # Combinar título y sinopsis
        title_plot = f"{data['title']} {data['plot']}"
        
        # Vectorizar el texto usando el vectorizador TF-IDF
        input_data = vectorizer.transform([title_plot]).toarray()

        # Hacer la predicción
        prediction = model.predict(input_data)
        
        # Convertir la predicción a una lista para que sea serializable en JSON
        prediction = prediction.tolist()
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
