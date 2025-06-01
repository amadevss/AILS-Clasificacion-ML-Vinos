from models.wine_classifier import WineClassifier
from utils.wine_features import WINE_FEATURES, WINE_CLASSES, validate_features
import numpy as np

def main():
    # Cargar el modelo entrenado
    classifier = WineClassifier()
    classifier.load_model()
    
    # Crear un ejemplo de vino para probar
    test_wine = {
        'Alcohol': 13.5,
        'Malic acid': 2.0,
        'Ash': 2.2,
        'Alcalinity of ash': 20.0,
        'Magnesium': 100.0,
        'Total phenols': 2.5,
        'Flavanoids': 2.8,
        'Nonflavanoid phenols': 0.3,
        'Proanthocyanins': 1.8,
        'Color intensity': 5.0,
        'Hue': 1.0,
        'OD280/OD315 of diluted wines': 2.5,
        'Proline': 800.0
    }
    
    # Validar las características
    is_valid, message = validate_features(test_wine)
    if not is_valid:
        print(f"Error: {message}")
        return
    
    # Convertir el diccionario a un array numpy
    features_array = np.array([[test_wine[feature] for feature in WINE_FEATURES]])
    
    # Realizar la predicción
    prediction, probabilities = classifier.predict(features_array)
    
    # Mostrar resultados
    print("\nCaracterísticas del vino:")
    for feature, value in test_wine.items():
        print(f"{feature}: {value}")
    
    print(f"\nPredicción: {WINE_CLASSES[prediction[0]]}")
    print("\nProbabilidades para cada clase:")
    for i, prob in enumerate(probabilities[0], 1):
        print(f"Clase {i}: {prob:.2%}")

if __name__ == "__main__":
    main() 