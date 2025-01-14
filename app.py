import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib


def batch_predict(file_path):
    data = pd.read_csv(file_path)
    data["Predicted_Price"] = np.random.rand(len(data)) * 100
    return data

with gr.Blocks() as app:
    gr.Markdown("# Prédiction des Prix des Voitures")
    
    model = joblib.load("model_lasso_Afdel_car_prices_predict.pkl")
    
    feature_columns = [
        "Kms_Driven", "Present_Price", "Fuel_Type", 
        "Seller_Type", "Transmission", "Age"
    ]

    def preprocess_input(inputs):
        input_data = pd.DataFrame([inputs], columns=feature_columns)
        fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
        seller_mapping = {"Dealer": 0, "Individual": 1}
        transmission_mapping = {"Manual": 0, "Automatic": 1}

        input_data["Fuel_Type"] = input_data["Fuel_Type"].map(fuel_mapping)
        input_data["Seller_Type"] = input_data["Seller_Type"].map(seller_mapping)
        input_data["Transmission"] = input_data["Transmission"].map(transmission_mapping)

        return input_data

    def predict_car_price(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
        inputs = [Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age]
        input_data = preprocess_input(inputs)
        prediction = model.predict(input_data)[0]
        return f"Prix estimé de la voiture : {prediction:.3f} $"

    with gr.Tab("Prédiction Individuelle"):
        gr.Markdown("### Entrez les caractéristiques de la voiture:")
        
        with gr.Row():
            kms_input = gr.Number(label="Kilomètres parcourus")
            price_input = gr.Number(label="Prix actuel (en $)")
            fuel_input = gr.Dropdown(["Petrol", "Diesel", "CNG"], label="Type de carburant")
            seller_input = gr.Dropdown(["Dealer", "Individual"], label="Type de vendeur")
            trans_input = gr.Dropdown(["Manual", "Automatic"], label="Transmission")
            age_input = gr.Number(label="Âge de la voiture (en années)")
        
            output_individual = gr.Textbox(label="Résultat")
            gr.Button("Prédire").click(predict_car_price,inputs=[kms_input, price_input, fuel_input,seller_input, trans_input, age_input],outputs=output_individual)

    with gr.Tab("Prédiction par Fichier"):
        gr.Markdown("### Générez ou téléchargez un fichier CSV avec les données des voitures:")
        
        file_upload = gr.File(label="Téléchargez votre fichier CSV", type="filepath")
        batch_predict_button = gr.Button("Prédire sur le fichier", variant="primary")
        
        batch_output = gr.Dataframe(label="Résultats des Prédictions")
        
        batch_predict_button.click(batch_predict,inputs=file_upload,outputs=batch_output)

app.launch()