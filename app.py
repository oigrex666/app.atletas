import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns
import pickle  
import os 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import base64

# Configuración de la página
st.set_page_config(page_title="Clasificador de Atletas", page_icon="🐦")


# CSS para cambiar el fondo
st.markdown("""
<style>
    .stApp {
        background-color: #17202a;
    }
    
    /* Estilo para las animaciones */
    .animation-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Función para mostrar animaciones de pájaros
def mostrar_animacion(tipo_atleta):
    animation_container = "<div class='animation-container'>"
    
    if tipo_atleta == "Fondista":
        # Animación de pájaro para fondista (águila planeando)
        animation_container += """
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjA3NzNlZDRiOWRhMWQ5ZjcwMWI5NzE0NmMwZGRjNDFiMTdhZGZmZiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/l0HlMWkHJ4UVVnL7W/giphy.gif" 
        width="300" alt="Pájaro de larga distancia">
        """
    else:  # Velocista
        # Animación de pájaro para velocista (colibrí rápido)
        animation_container += """
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjM0MDhmZjlkZmRjMTk2ZTUwMGI1MGUyOWJkYjZlNzBkZTNkMGI3YyZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/3oEduLr20oZ0GVijkY/giphy.gif" 
        width="300" alt="Pájaro veloz">
        """
    
    animation_container += "</div>"
    return animation_container

# Función para cargar los datos
def cargar_datos():
    try:
        df = pd.read_csv('atletas.csv')
        return df
    except:
        st.error("No se pudo cargar el archivo de datos")
        return None

# Barra lateral
st.sidebar.title("Menú de Navegación")
pagina = st.sidebar.selectbox("Selecciona una opción:", ["home","Preprocesamiento","Predicción", "Modelo", "Datos", "Métricas"])

# Variables en la barra lateral
st.sidebar.subheader("Variables de entrada")
edad = st.sidebar.slider("Edad", 15, 60, 25)
frecuencia = st.sidebar.slider("Frecuencia Cardíaca (lpm)", 40, 100, 70)
volumen = st.sidebar.slider("Volumen Sistólico (ml)", 50, 200, 75)

# Hiperparámetros en la barra lateral
st.sidebar.subheader("Hiperparámetros del modelo")
max_depth = st.sidebar.slider("Profundidad máxima del árbol", 1, 4, 2)
criterion = st.sidebar.selectbox("Criterio de división", ["gini", "entropy"])

# Carga de datos
df = cargar_datos()

if pagina == "Datos":
    st.header("Datos de Atletas")
    if df is not None:
        st.write("Vista previa de los datos:")
        st.dataframe(df)
        st.subheader("Distribución por clase")
        fig, ax = plt.subplots()
        df['Clasificación'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

elif pagina == "Preprocesamiento":
    st.header("Preprocesamiento de Datos")
    if df is not None:
        st.write("Datos originales:")
        st.dataframe(df.head())

        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        st.write("Datos después de la normalización:")
        st.dataframe(df_scaled.head())
    else:
        st.warning("No hay datos disponibles para preprocesar.")

elif pagina == "Modelo":
    st.header("Entrenamiento del Modelo")
    if df is not None:
        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        y = df['Clasificación']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if st.button("Entrenar Modelo"):
            with st.spinner("Entrenando..."):
                modelo = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
                modelo.fit(X_train, y_train)
                os.makedirs('modelo', exist_ok=True)
                with open('modelo/clasificador.pkl', 'wb') as f:
                    pickle.dump(modelo, f)
                with open('modelo/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                y_pred = modelo.predict(X_test)
                precision = accuracy_score(y_test, y_pred)
                st.success(f"¡Modelo entrenado! Precisión: {precision:.2f}")

elif pagina == "Predicción":
    st.header("Hacer Predicción")
    if os.path.exists('modelo/clasificador.pkl'):
        with open('modelo/clasificador.pkl', 'rb') as f:
            modelo = pickle.load(f)
        with open('modelo/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        dato = [[edad, frecuencia, volumen]]
        dato_scaled = scaler.transform(dato)
        prediccion = modelo.predict(dato_scaled)[0]
        
        st.success(f"Predicción: {prediccion}")
        
        # Mostrar animación según la predicción
        st.markdown(mostrar_animacion(prediccion), unsafe_allow_html=True)
        
        # Descripción del pájaro según el tipo de atleta
        if prediccion == "Fondista":
            st.info("Como un águila que planea en las corrientes térmicas, este atleta tiene resistencia y eficiencia para distancias largas.")
        else:
            st.info("Como un colibrí en pleno vuelo, este atleta posee velocidad explosiva y rápidos reflejos.")
        
        # Mostrar detalles adicionales
        st.subheader("Detalles de la predicción:")
        probabilidades = modelo.predict_proba(dato_scaled)[0]
        st.write("Probabilidad por clase:")
        for i, prob in enumerate(probabilidades):
            st.write(f"Clase {modelo.classes_[i]}: {prob:.2f}")
            
        # Características del atleta
        st.subheader("Características del atleta:")
        st.write(f"Edad: {edad} años")
        st.write(f"Frecuencia cardíaca: {frecuencia} lpm")
        st.write(f"Volumen sistólico: {volumen} ml")
        
    else:
        st.warning("No hay modelo entrenado. Ve a la página 'Modelo' para entrenarlo primero.")

elif pagina == "Métricas":
    st.header("Métricas del Modelo")
    if df is not None:
        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        y = df['Clasificación']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        modelo = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.write(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precisión (Precision Score): {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

        st.subheader(f"Matriz de Confusión (Profundidad del Árbol: {modelo.get_depth()})")
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(y.unique())
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, xticklabels=labels, yticklabels=labels)
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Real")
        st.pyplot(fig_cm)

        st.subheader("Visualización del Árbol de Decisión")
        fig_tree, ax_tree = plt.subplots(figsize=(10, 5))
        plot_tree(modelo, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())], filled=True, ax=ax_tree)
        st.pyplot(fig_tree)

        st.write("----")
        st.write("📈 Comparación con Regresión Logística:")
        st.write(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision Score: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    else:
        st.warning("No hay modelo entrenado o datos disponibles.")
elif pagina=='home':
    st.title('Inicio')
    st.write('Esta app te permite predecir si alguien es fondista o velocista en función de las variables edad, Frecuencia Cardíaca y Volumen Sistólico de la persona.')
    
    # Añadir una pequeña presentación sobre la analogía con pájaros
    st.subheader("Rendimiento Deportivo y Aves")
    st.write("""
    Este clasificador ayuda a determinar la modalidad deportiva más adecuada para atletas 
    basándose en parámetros fisiológicos clave, asociando cada tipo con un ave representativa:
    
    - **Fondistas** 🦅: Como las águilas que planean largas distancias con eficiencia energética
    - **Velocistas** 🐦: Como los colibríes, capaces de movimientos explosivos y rápidos
    
    Navega a la sección 'Predicción' para clasificar a un atleta según sus características.
    """)
    
    # Mostrar ambas animaciones en la página de inicio
    st.markdown("<h3 style='text-align: center;'>Analogías en la naturaleza</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='text-align: center;'>Fondista - Como un águila</h4>", unsafe_allow_html=True)
        st.markdown(mostrar_animacion("Fondista"), unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h4 style='text-align: center;'>Velocista - Como un colibrí</h4>", unsafe_allow_html=True)
        st.markdown(mostrar_animacion("Velocista"), unsafe_allow_html=True)
