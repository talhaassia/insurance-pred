from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('modelreg.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('template.html')  # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données d'entrée
    data = request.get_json()

    # Transformer les données en DataFrame
    df = pd.DataFrame(data, index=[0])

    # Standardiser les données non-binaires avec le scaler
    non_binary_columns = [col for col in df.columns if col not in ['sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']]
    df_non_binary = df[non_binary_columns]
    df_non_binary_scaled = scaler.transform(df_non_binary)
    df_non_binary_scaled = pd.DataFrame(df_non_binary_scaled, columns=non_binary_columns, index=df.index)

    # Fusionner les données standardisées et binaires
    binary_columns = [col for col in df.columns if col not in non_binary_columns]
    df_binary = df[binary_columns]
    Xs_final = pd.concat([df_non_binary_scaled, df_binary], axis=1)

    # Faire une prédiction
    prediction = model.predict(Xs_final)

    # Retourner la prédiction sous forme de JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
