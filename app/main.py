from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from models.wine_classifier import WineClassifier
from utils.wine_features import WINE_FEATURES, WINE_CLASSES, validate_features
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif
import lazypredict
from lazypredict.Supervised import LazyClassifier
import mlflow
import mlflow.sklearn

app = FastAPI(title="Clasificador de Vinos")

# Montar archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar el modelo
classifier = WineClassifier()
classifier.load_model()

# Configurar estilo de gráficos
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Cargar el dataset de vinos
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Usar LazyPredict para comparar modelos
mlflow.set_experiment('Wine_Classification')
with mlflow.start_run(run_name='LazyPredict_Backend'):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)
    for model_name, row in models.iterrows():
        mlflow.log_metric(f'{model_name}_Accuracy', row['Accuracy'])
        mlflow.log_metric(f'{model_name}_F1_Score', row['F1 Score'])

# Calcular ganancia de información mutua
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# Función para generar gráficos
def generate_plot():
    # Histogramas de las variables
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(X.columns):
        plt.subplot(4, 4, i+1)
        sns.histplot(X[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

# Función para generar matriz de correlación
def generate_correlation_matrix():
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

# Función para generar gráfico de ganancia de información mutua
def generate_mi_scores():
    plt.figure(figsize=(10, 6))
    mi_scores.plot(kind='bar')
    plt.title('Ganancia de Información Mutua')
    plt.xlabel('Características')
    plt.ylabel('Ganancia de Información')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal con el formulario de entrada y análisis exploratorio de datos"""
    # Generar gráficos
    histograms = generate_plot()
    correlation_matrix = generate_correlation_matrix()
    mi_scores_plot = generate_mi_scores()
    
    # Convertir el DataFrame de modelos a un diccionario
    models_dict = {}
    for index, row in models.iterrows():
        models_dict[index] = {
            'Accuracy': row['Accuracy'],
            'F1 Score': row['F1 Score']
        }
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": WINE_FEATURES,
            "histograms": histograms,
            "correlation_matrix": correlation_matrix,
            "mi_scores_plot": mi_scores_plot,
            "mi_scores": mi_scores.to_dict(),
            "models": models_dict
        }
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    alcohol: float = Form(...),
    malic_acid: float = Form(...),
    ash: float = Form(...),
    alcalinity_of_ash: float = Form(...),
    magnesium: float = Form(...),
    total_phenols: float = Form(...),
    flavanoids: float = Form(...),
    nonflavanoid_phenols: float = Form(...),
    proanthocyanins: float = Form(...),
    color_intensity: float = Form(...),
    hue: float = Form(...),
    od280_od315: float = Form(...),
    proline: float = Form(...)
):
    """Endpoint para realizar predicciones"""
    # Crear diccionario con las características
    features = {
        'Alcohol': alcohol,
        'Malic acid': malic_acid,
        'Ash': ash,
        'Alcalinity of ash': alcalinity_of_ash,
        'Magnesium': magnesium,
        'Total phenols': total_phenols,
        'Flavanoids': flavanoids,
        'Nonflavanoid phenols': nonflavanoid_phenols,
        'Proanthocyanins': proanthocyanins,
        'Color intensity': color_intensity,
        'Hue': hue,
        'OD280/OD315 of diluted wines': od280_od315,
        'Proline': proline
    }
    
    # Validar características
    is_valid, message = validate_features(features)
    if not is_valid:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "features": WINE_FEATURES,
                "error": message
            }
        )
    
    # Convertir a array numpy
    features_array = np.array([[features[feature] for feature in WINE_FEATURES]])
    
    # Realizar predicción
    prediction, probabilities = classifier.predict(features_array)
    
    # Preparar resultados
    result = {
        "prediction": WINE_CLASSES[prediction[0]],
        "probabilities": {
            f"Clase {i+1}": f"{prob:.2%}"
            for i, prob in enumerate(probabilities[0])
        },
        "features": features
    }
    
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result
        }
    ) 