import numpy as np 
import pandas as pd  
import streamlit as st  
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
import pickle 
import os 

def carga_datos():
    """
    Función para cargar los datos desde un archivo CSV.
    
    Returns:
        DataFrame: El DataFrame cargado o None si ocurre un error.
    """
    try:
        # Intenta cargar el archivo CSV
        df = pd.read_csv('data/atletas.csv')
        return df
    except FileNotFoundError:
        # Maneja el error si el archivo no existe
        print("Error: No se encontró el archivo 'data/atletas.csv'")
        return None
    except Exception as e:
        # Maneja cualquier otro error que pueda ocurrir
        print(f"Error al cargar los datos: {e}")
        return None

def crear_modelo(df):
    """
    Función para crear, entrenar y evaluar un modelo de regresión logística.
    
    Args:
        df (DataFrame): El DataFrame que contiene los datos.
        
    Returns:
        tuple: (modelo entrenado, escalador) o (None, None) si ocurre un error.
    """
    if df is None:
        return None, None
        
    try:
        # Examina valores únicos en la columna de clasificación (para depuración)
        print("Valores únicos en 'Clasificación':", df['Clasificación'].unique())
        
        # Convierte la variable objetivo a valores numéricos si son categóricos
        if df['Clasificación'].dtype == 'object':  # Verifica si los datos son categóricos (tipo objeto)
            label_encoder = LabelEncoder()  # Crea un codificador de etiquetas
            df['Clasificación_num'] = label_encoder.fit_transform(df['Clasificación'])  # Transforma etiquetas a números
            # Guarda el mapeo de categorías a números para referencia
            mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            print("Mapeo de categorías:", mapping)
            y = df[['Clasificación_num']]  # Variable objetivo codificada
        else:
            y = df[['Clasificación']]  # Variable objetivo ya es numérica
        
        # Selecciona las características (variables independientes)
        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]

        # Escala las características para mejorar el rendimiento del modelo
        scaler_lv = StandardScaler()  # Crea un escalador
        X_scaled = scaler_lv.fit_transform(X)  # Ajusta y transforma los datos

        # Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Crea y entrena el modelo de regresión logística
        model_lv = LogisticRegression(max_iter=1000)  # Aumenta el número máximo de iteraciones
        model_lv.fit(X_train, y_train.values.ravel())  # Entrena el modelo (ravel convierte a array 1D)

        # Evalúa el modelo con el conjunto de prueba
        y_pred = model_lv.predict(X_test)  # Hace predicciones
        accuracy = model_lv.score(X_test, y_test)  # Calcula la precisión

        # Muestra métricas de evaluación
        print("\nMétricas de evaluación del modelo:")
        print(f"Precisión (accuracy): {accuracy:.4f}")

        # Prueba el modelo con un ejemplo
        input_dict = {'Edad': 25, 'Frecuencia Cardiaca Basal (lpm)': 70, 'Volumen Sistólico (ml)': 75}
        X_in = np.array(list(input_dict.values())).reshape(1, -1)  # Convierte el diccionario a array
        X_in = scaler_lv.transform(X_in)  # Aplica el mismo escalado usado en entrenamiento
        prediccion = model_lv.predict(X_in)  # Hace la predicción
        probabilidades = model_lv.predict_proba(X_in)  # Calcula probabilidades para cada clase
        
        # Muestra la predicción (convirtiendo de vuelta a la categoría original si es necesario)
        if df['Clasificación'].dtype == 'object':
            categoria_predicha = label_encoder.inverse_transform(prediccion)[0]  # Convierte número a categoría
            print(f"\nPredicción para {input_dict}: {categoria_predicha}")
        else:
            print(f"\nPredicción para {input_dict}: {prediccion[0]}")
            
        return model_lv, scaler_lv  # Devuelve el modelo y el escalador
    except Exception as e:
        # Maneja cualquier error durante la creación del modelo
        print(f"Error al crear el modelo: {e}")
        return None, None

def main():
    """
    Función principal que ejecuta todo el proceso.
    """
    print("Cargando datos...")
    df = carga_datos()  # Carga los datos
    
    if df is not None:
        print(f"Datos cargados correctamente. Shape: {df.shape}")
        print("\nCreando y evaluando modelo...")
        model_lv, scaler_lv = crear_modelo(df)  # Crea y evalúa el modelo
        
        if model_lv is not None and scaler_lv is not None:
            try:
                # Crea directorio para guardar el modelo si no existe
                os.makedirs('app_1v', exist_ok=True)
                
                # Guarda el modelo entrenado en un archivo usando pickle
                with open('app_1v/model_lv.pkl', 'wb') as f:
                    pickle.dump(model_lv, f)
                
                # Guarda el escalador en un archivo usando pickle
                with open('app_1v/scaler_lv.pkl', 'wb') as g:
                    pickle.dump(scaler_lv, g)
                    
                print("\nModelo y scaler guardados correctamente en la carpeta 'app_1v'")
            except Exception as e:
                # Maneja errores al guardar el modelo
                print(f"Error al guardar el modelo: {e}")

# Punto de entrada del script
if __name__ == "__main__":
    main()  
