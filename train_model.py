from models.wine_classifier import WineClassifier
import os

def main():
    # Crear directorio models si no existe
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Inicializar y entrenar el clasificador
    classifier = WineClassifier()
    accuracy, report = classifier.train()
    
    # Guardar el modelo y el scaler
    classifier.save_model()
    
    print("\nModelo entrenado y guardado exitosamente!")
    print(f"El modelo se guardó en: {os.path.abspath('models/wine_model.joblib')}")
    print(f"El scaler se guardó en: {os.path.abspath('models/scaler.joblib')}")

if __name__ == "__main__":
    main() 