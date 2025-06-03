import numpy as np 
import pandas as pd  
import streamlit as st  
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
import pickle 
import os 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

def carga_datos():
    try:
        df = pd.read_csv('atletas.csv')
        return df
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'atletas.csv'")
        return None
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def crear_red_neuronal(df):
    if df is None:
        return None, None

    try:
        print("Valores únicos en 'Clasificación':", df['Clasificación'].unique())

        if df['Clasificación'].dtype == 'object':
            label_encoder = LabelEncoder()
            df['Clasificación_num'] = label_encoder.fit_transform(df['Clasificación'])
            mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            print("Mapeo de categorías:", mapping)
            y = df['Clasificación_num']
        else:
            y = df['Clasificación']

        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        scaler_lv = StandardScaler()
        X_scaled = scaler_lv.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        num_classes = len(np.unique(y_train))
        is_binary = num_classes == 2

        # Red neuronal
        model_nn = Sequential()
        model_nn.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
        model_nn.add(Dense(8, activation='relu'))
        model_nn.add(Dense(1 if is_binary else num_classes, activation='sigmoid' if is_binary else 'softmax'))

        model_nn.compile(
            loss='binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        model_nn.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

        loss, accuracy = model_nn.evaluate(X_test, y_test, verbose=0)
        print(f"\nPrecisión del modelo de red neuronal: {accuracy:.4f}")

        # Prueba
        input_dict = {'Edad': 25, 'Frecuencia Cardiaca Basal (lpm)': 70, 'Volumen Sistólico (ml)': 75}
        X_in = np.array(list(input_dict.values())).reshape(1, -1)
        X_in = scaler_lv.transform(X_in)
        pred = model_nn.predict(X_in)

        if is_binary:
            pred_class = int(pred[0] > 0.5)
        else:
            pred_class = np.argmax(pred, axis=1)[0]

        if df['Clasificación'].dtype == 'object':
            categoria_predicha = label_encoder.inverse_transform([pred_class])[0]
            print(f"\nPredicción para {input_dict}: {categoria_predicha}")
        else:
            print(f"\nPredicción para {input_dict}: {pred_class}")

        return model_nn, scaler_lv
    except Exception as e:
        print(f"Error al crear la red neuronal: {e}")
        return None, None

def main():
    print("Cargando datos...")
    df = carga_datos()

    if df is not None:
        print(f"Datos cargados correctamente. Shape: {df.shape}")
        print("\nCreando y evaluando modelo de red neuronal...")
        model_nn, scaler_lv = crear_red_neuronal(df)

        if model_nn is not None and scaler_lv is not None:
            try:
                os.makedirs('app_1v', exist_ok=True)
                model_nn.save('app_1v/model_nn.h5')
                with open('app_1v/scaler_lv.pkl', 'wb') as g:
                    pickle.dump(scaler_lv, g)
                print("\nModelo y scaler guardados correctamente en 'app_1v'")
            except Exception as e:
                print(f"Error al guardar el modelo: {e}")

if __name__ == "__main__":
    main()
