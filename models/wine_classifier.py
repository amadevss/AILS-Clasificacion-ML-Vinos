import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from ucimlrepo import fetch_ucirepo

class WineClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Carga el dataset de vinos desde UCI"""
        wine = fetch_ucirepo(id=109)
        X = wine.data.features
        y = wine.data.targets
        return X, y
    
    def preprocess_data(self, X, y):
        """Preprocesa los datos para el entrenamiento"""
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar las características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self):
        """Entrena el modelo con los datos de vinos"""
        # Cargar y preprocesar datos
        X, y = self.load_data()
        X_train_scaled, X_test_scaled, y_train, y_test = self.preprocess_data(X, y)
        
        # Entrenar el modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar el modelo
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Precisión del modelo: {accuracy:.2f}")
        print("\nReporte de clasificación:")
        print(report)
        
        return accuracy, report
    
    def predict(self, features):
        """Realiza predicciones con el modelo entrenado"""
        # Escalar las características
        features_scaled = self.scaler.transform(features)
        # Realizar predicción
        prediction = self.model.predict(features_scaled)
        # Obtener probabilidades
        probabilities = self.model.predict_proba(features_scaled)
        
        return prediction, probabilities
    
    def save_model(self, model_path='models/wine_model.joblib', scaler_path='models/scaler.joblib'):
        """Guarda el modelo y el scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path='models/wine_model.joblib', scaler_path='models/scaler.joblib'):
        """Carga el modelo y el scaler guardados"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) 