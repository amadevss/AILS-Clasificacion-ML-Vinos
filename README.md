# Clasificador de Vinos con Análisis Exploratorio

Este proyecto es una aplicación web que utiliza FastAPI para realizar análisis exploratorio de datos y clasificación de vinos utilizando el dataset de UCI Wine. La aplicación incluye visualizaciones interactivas y un modelo de clasificación para predecir la clase de vino basado en sus características químicas.

## Características

- Análisis Exploratorio de Datos (EDA) con visualizaciones:
  - Histogramas de distribución de variables
  - Matriz de correlación
  - Análisis de ganancia de información mutua
- Comparación automática de múltiples modelos de clasificación
- Interfaz web interactiva para realizar predicciones
- Visualización de resultados en tiempo real

## Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Crear y activar un entorno virtual:
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
.
├── app/
│   ├── main.py              # Aplicación principal FastAPI
│   ├── models/
│   │   └── wine_classifier.py
│   └── utils/
│       └── wine_features.py
├── static/
│   └── css/
│       └── styles.css
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
├── requirements.txt
└── README.md
```

## Uso

1. Iniciar el servidor:
```bash
uvicorn app.main:app --reload
```

2. Abrir el navegador y acceder a:
```
http://localhost:8000
```

3. En la página principal podrás ver:
   - Análisis exploratorio de datos con visualizaciones
   - Comparación de modelos de clasificación
   - Formulario para realizar predicciones

4. Para realizar una predicción:
   - Completa el formulario con los valores de las características del vino
   - Haz clic en "Clasificar Vino"
   - Verás el resultado de la clasificación y las probabilidades para cada clase

## Uso de Jupyter Notebook

1. Inicia Jupyter Notebook o JupyterLab en la raíz del proyecto:
```bash
jupyter notebook
```

2. Abre el archivo `wine_analysis.ipynb`.
3. Ejecuta las celdas en orden para:
   - Cargar y explorar el dataset de vinos
   - Visualizar histogramas y matriz de correlación
   - Calcular y graficar la ganancia de información mutua
   - Dividir y escalar los datos
   - Comparar modelos con LazyPredict
   - Registrar automáticamente los resultados en MLflow

4. Puedes modificar el notebook para probar tus propios modelos, análisis o visualizaciones adicionales.

## Uso y Visualización de Experimentos con MLflow

### Instalación de MLflow
Si no tienes MLflow instalado, puedes instalarlo con:
```bash
pip install mlflow
```

### Cómo iniciar el servidor de MLflow
1. En la raíz del proyecto, ejecuta:
```bash
mlflow ui
```

2. Abre tu navegador y accede a:
```
http://127.0.0.1:5000
```

### ¿Qué verás en MLflow?
- Un experimento llamado `Wine_Classification`.
- Runs (ejecuciones) con las métricas de los modelos probados tanto desde el backend como desde el notebook.
- Podrás comparar métricas como Accuracy y F1 Score entre diferentes modelos y ejecuciones.

### Buenas prácticas
- Ejecuta el notebook y el backend con el servidor de MLflow corriendo para registrar todos los experimentos.
- Usa MLflow para comparar resultados de diferentes configuraciones o modelos.
- Puedes agregar parámetros y artefactos personalizados a los runs de MLflow si lo deseas.

## Características del Vino

El modelo utiliza las siguientes características químicas:
- Alcohol
- Ácido málico
- Ceniza
- Alcalinidad de la ceniza
- Magnesio
- Fenoles totales
- Flavonoides
- Fenoles no flavonoides
- Proantocianinas
- Intensidad del color
- Tono
- OD280/OD315 de vinos diluidos
- Prolina

## Tecnologías Utilizadas

- FastAPI: Framework web de alto rendimiento
- Pandas: Manipulación y análisis de datos
- NumPy: Computación numérica
- Scikit-learn: Machine Learning
- Matplotlib y Seaborn: Visualización de datos
- LazyPredict: Comparación automática de modelos
- Jinja2: Motor de plantillas
- Bootstrap: Framework CSS para el diseño web

## Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz un Fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Uso con Docker Compose

Puedes levantar todos los servicios (FastAPI, MLflow y Jupyter) fácilmente usando Docker Compose.

1. Asegúrate de haber construido y subido tu imagen a Docker Hub:
   ```bash
   docker build -t brychxpin/ailabschool-p1:latest .
   docker push brychxpin/ailabschool-p1:latest
   ```

2. Levanta todos los servicios:
   ```bash
   docker-compose up
   ```
   (O en modo background: `docker-compose up -d`)

3. Accede a:
   - FastAPI: [http://localhost:8000](http://localhost:8000)
   - MLflow: [http://localhost:5000](http://localhost:5000)
   - Jupyter: [http://localhost:8888](http://localhost:8888)

---

## Despliegue en la nube (Deploy)

Puedes desplegar tu proyecto en la nube usando la imagen de Docker que creaste. Aquí tienes los pasos generales y recomendaciones:

### 1. Sube tu imagen a Docker Hub

Si no lo has hecho, sube tu imagen:
```bash
docker login
# Luego:
docker push brychxpin/ailabschool-p1:latest
```

### 2. Opciones de despliegue

- **Railway, Render, Heroku (con Docker):**
  - Solo necesitas tu cuenta y conectar tu repositorio o imagen de Docker.
  - Sigue las instrucciones de cada plataforma para crear un servicio web con Docker Compose o un solo contenedor.

- **AWS ECS / Fargate:**
  - Sube tu imagen a Docker Hub o Amazon ECR.
  - Crea un servicio ECS usando tu imagen y expón los puertos necesarios.

- **Google Cloud Run / GKE:**
  - Sube tu imagen a Google Container Registry o Artifact Registry.
  - Despliega usando Cloud Run (para apps web) o GKE (para orquestación avanzada).

- **Azure Container Instances / Web Apps:**
  - Sube tu imagen a Azure Container Registry o usa Docker Hub.
  - Crea una instancia de contenedor o un Web App con tu imagen.

### 3. Recomendaciones
- Asegúrate de exponer los puertos necesarios (8000, 5000, 8888) en la configuración del servicio.
- Usa variables de entorno para credenciales y configuraciones sensibles.
- Si usas volúmenes, configura almacenamiento persistente en la nube.
- Consulta la documentación oficial de cada proveedor para detalles específicos.

---

¿Dudas? ¡Abre un issue o pregunta en la sección de discusiones del repositorio! 