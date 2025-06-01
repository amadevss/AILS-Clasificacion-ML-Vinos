# Nombres de las características del vino
WINE_FEATURES = [
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]

# Nombres de las clases de vino
WINE_CLASSES = {
    1: "Vino de la clase 1",
    2: "Vino de la clase 2",
    3: "Vino de la clase 3"
}

def get_feature_ranges():
    """Retorna los rangos típicos para cada característica del vino"""
    return {
        'Alcohol': (11.0, 15.0),
        'Malic acid': (0.7, 6.0),
        'Ash': (1.3, 3.2),
        'Alcalinity of ash': (10.0, 30.0),
        'Magnesium': (70.0, 160.0),
        'Total phenols': (0.9, 3.9),
        'Flavanoids': (0.3, 5.1),
        'Nonflavanoid phenols': (0.1, 0.7),
        'Proanthocyanins': (0.4, 3.6),
        'Color intensity': (1.3, 13.0),
        'Hue': (0.5, 1.7),
        'OD280/OD315 of diluted wines': (1.2, 4.0),
        'Proline': (278.0, 1680.0)
    }

def validate_features(features):
    """Valida que las características estén dentro de rangos razonables"""
    ranges = get_feature_ranges()
    for feature, value in features.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if not (min_val <= value <= max_val):
                return False, f"El valor de {feature} debe estar entre {min_val} y {max_val}"
    return True, "Todas las características son válidas" 