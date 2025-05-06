import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Enfermedad Cardíaca",
    page_icon="❤️",
    layout="wide"
)

# Título principal
st.title("❤️ Predictor de Enfermedad Cardíaca")
st.markdown("""
Esta aplicación utiliza modelos de machine learning (AdaBoost, Gradient Boosting y XGBoost) 
para predecir la presencia de enfermedad cardíaca basada en factores de riesgo y datos clínicos.
""")

# Sidebar para navegación y selección de modelos
st.sidebar.title("Configuración")

# Selección de modelo
selected_model = st.sidebar.selectbox(
    "Selecciona un modelo",
    ["AdaBoost", "Gradient Boosting", "XGBoost"]
)

# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        # Intentamos cargar todos los modelos
        with open('best_models.pkl', 'rb') as file:
            models = pickle.load(file)
        st.sidebar.success("✅ Modelos cargados correctamente")
        return models
    except Exception as e:
        st.sidebar.error(f"⚠️ Error al cargar modelos: {e}")
        st.sidebar.info("Por favor, asegúrate de guardar tus modelos primero")
        return None

# Función para cargar datos
@st.cache_data
def load_data():
    try:
        # Definir columnas
        columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        
        # Cargar los tres datasets
        df1 = pd.read_csv('processed.cleveland.data', names=columnas)
        df2 = pd.read_csv('processed.switzerland.data', names=columnas)
        df3 = pd.read_csv('processed.hungarian.data', names=columnas)
        
        # Combinar datasets
        df = pd.concat([df1, df2, df3], ignore_index=True)
        
        # Preprocesamiento básico
        # Reemplazamos valores faltantes (?)
        df = df.replace('?', np.nan)
        
        # Convertimos columnas a numéricas
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convertimos la variable objetivo en binaria (0 = no enfermedad, 1 = enfermedad)
        df['target'] = (df['num'] > 0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

def prepare_input_for_model(input_data, model):
    """
    Prepara los datos de entrada para el modelo, incluyendo la codificación de variables categóricas.
    """
    # Convertir el diccionario de entrada a un DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Codificación one-hot para 'cp' y 'restecg'
    cp_dummies = pd.get_dummies(input_df['cp'], prefix='cp', drop_first=True)
    restecg_dummies = pd.get_dummies(input_df['restecg'], prefix='restecg', drop_first=True)
    
    # Concatenar las columnas dummy al DataFrame original
    input_df = pd.concat([input_df, cp_dummies, restecg_dummies], axis=1)
    
    # Eliminar las columnas originales 'cp' y 'restecg'
    input_df = input_df.drop(['cp', 'restecg'], axis=1)
    
    # Seleccionar solo las características que el modelo espera, en el orden correcto
    expected_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 
                         'cp_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'restecg_0.0', 'restecg_1.0', 'restecg_2.0']
    
    # Asegurar que estén todas las columnas
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_features]
    
    return input_df

# Definiciones para las características
def get_feature_descriptions():
    return {
        'age': 'Edad en años',
        'sex': 'Sexo (1 = masculino; 0 = femenino)',
        'trestbps': 'Presión arterial en reposo (mm Hg)',
        'chol': 'Colesterol sérico (mg/dl)',
        'fbs': 'Glucemia en ayunas > 120 mg/dl (1 = verdadero; 0 = falso)',
        'thalach': 'Frecuencia cardíaca máxima alcanzada',
        'exang': 'Angina inducida por ejercicio (1 = sí; 0 = no)',
        'oldpeak': 'Depresión del ST inducida por el ejercicio',
        'cp': 'Tipo de dolor de pecho (1-4)',
        'restecg': 'Resultados electrocardiográficos en reposo (0-2)',
        'cp_1.0': 'Tipo de dolor de pecho: Angina típica',
        'cp_2.0': 'Tipo de dolor de pecho: Angina atípica',
        'cp_3.0': 'Tipo de dolor de pecho: Dolor no anginoso',
        'cp_4.0': 'Tipo de dolor de pecho: Asintomático',
        'restecg_0.0': 'ECG en reposo: Normal',
        'restecg_1.0': 'ECG en reposo: Anomalía del ST-T',
        'restecg_2.0': 'ECG en reposo: Hipertrofia ventricular izquierda'
    }

# Rangos para los sliders
def get_feature_ranges():
    return {
        'age': (25, 80, 50),
        'sex': (0, 1, 1),
        'trestbps': (90, 200, 120),
        'chol': (120, 570, 230),
        'fbs': (0, 1, 0),
        'thalach': (70, 210, 150),
        'exang': (0, 1, 0),
        'oldpeak': (0.0, 6.0, 1.0),
        'cp': (1, 4, 1),  # Rango original para cp
        'restecg': (0, 2, 0)  # Rango original para restecg
    }

# Intentar cargar los modelos y datos
models = load_models()
df = load_data()
feature_descriptions = get_feature_descriptions()
feature_ranges = get_feature_ranges()

# Definir las características que usamos para predicción
feature_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'cp_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'restecg_0.0', 'restecg_1.0', 'restecg_2.0']


# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Predicción Individual", "Predicción por Lotes", "Rendimiento de Modelos"])

with tab1:
    st.header("Exploración de Datos")
    
    if df is not None:
        st.write(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas")
        
        # Mostrar datos
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        st.write(df.describe())
        
        # Distribución de enfermedad cardíaca
        st.subheader("Distribución de enfermedad cardíaca")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='target', data=df, ax=ax)
        ax.set_xlabel('Enfermedad Cardíaca')
        ax.set_ylabel('Cantidad')
        ax.set_xticklabels(['No', 'Sí'])
        st.pyplot(fig)
        
        # Correlaciones
        st.subheader("Matriz de correlación")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        st.pyplot(fig)
        
        # Gráficos de distribución por edad y sexo
        st.subheader("Distribución por edad y sexo")
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distribución por edad
        sns.histplot(data=df, x='age', hue='target', multiple='stack', ax=ax[0])
        ax[0].set_title('Distribución por edad')
        
        # Distribución por sexo
        df_sex = df.groupby(['sex', 'target']).size().reset_index(name='count')
        sns.barplot(x='sex', y='count', hue='target', data=df_sex, ax=ax[1])
        ax[1].set_title('Distribución por sexo')
        ax[1].set_xticklabels(['Femenino', 'Masculino'])
        
        st.pyplot(fig)
    else:
        st.warning("No se pudieron cargar los datos. Asegúrate de tener los archivos de datos en el mismo directorio.")

with tab2:
    st.header("Predicción Individual")
    
    if models is None:
        st.warning("No se pudieron cargar los modelos. Por favor, guarda tus modelos primero.")
    else:
        st.write(f"Usando el modelo: **{selected_model}**")
        
        # Crear entradas para cada característica
        st.subheader("Ingresa los valores para hacer una predicción:")
        
        # Creamos columnas para organizar mejor los inputs
        col1, col2 = st.columns(2)
        
        # Diccionario para almacenar los valores de entrada
        input_data = {}
        
        # Creamos inputs dinámicamente basados en las características
        input_data['age'] = col1.slider(feature_descriptions['age'], *feature_ranges['age'])
        input_data['sex'] = col1.radio(feature_descriptions['sex'], options=[0, 1], horizontal=True)
        input_data['trestbps'] = col1.slider(feature_descriptions['trestbps'], *feature_ranges['trestbps'])
        input_data['chol'] = col1.slider(feature_descriptions['chol'], *feature_ranges['chol'])
        input_data['fbs'] = col1.radio(feature_descriptions['fbs'], options=[0, 1], horizontal=True)
        input_data['thalach'] = col1.slider(feature_descriptions['thalach'], *feature_ranges['thalach'])
        input_data['exang'] = col2.radio(feature_descriptions['exang'], options=[0, 1], horizontal=True)
        input_data['oldpeak'] = col2.slider(feature_descriptions['oldpeak'], *feature_ranges['oldpeak'], step=0.1)
        input_data['cp'] = col2.selectbox(feature_descriptions['cp'], options=list(range(feature_ranges['cp'][0], feature_ranges['cp'][1] + 1)))
        input_data['restecg'] = col2.selectbox(feature_descriptions['restecg'], options=list(range(feature_ranges['restecg'][0], feature_ranges['restecg'][1] + 1)))
        
        # Botón para realizar la predicción
        if st.button("Realizar Predicción"):
            # Preparar los datos para la predicción
            input_df = prepare_input_for_model(input_data, models[selected_model])
            
            # Realizar la predicción con el modelo seleccionado
            model = models[selected_model]
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            # Mostrar resultados
            st.subheader("Resultado de la Predicción:")
            
            # Determinar la clase y mostrar resultado
            if prediction == 1:
                st.error("⚠️ Riesgo de enfermedad cardíaca: **POSITIVO**")
                st.write("El modelo predice que hay riesgo de enfermedad cardíaca.")
            else:
                st.success("✅ Riesgo de enfermedad cardíaca: **NEGATIVO**")
                st.write("El modelo predice que no hay riesgo de enfermedad cardíaca.")
            
            # Mostrar probabilidades
            st.subheader("Probabilidades:")
            if len(probabilities) == 2:
                prob_df = pd.DataFrame({
                    'Clase': ['Sin enfermedad', 'Con enfermedad'],
                    'Probabilidad': probabilities
                })

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Clase', y='Probabilidad', data=prob_df, palette="viridis", ax=ax)
                ax.set_ylim(0, 1)
                ax.set_title("Probabilidades de cada clase")
                st.pyplot(fig)
            else:
                st.warning("⚠️ El modelo no devolvió dos probabilidades. Resultado inesperado:")
                st.write(probabilities)
            
            
            # Gráfico de barras para las probabilidades
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Clase', y='Probabilidad', data=prob_df, palette="viridis", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Probabilidades de cada clase")
            st.pyplot(fig)
            
            # Mostrar factores más importantes
            if hasattr(model, 'feature_importances_'):
                st.subheader("Factores más importantes para este modelo:")
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Intersección de las características del modelo y las esperadas
                available_features = [f for f in feature_names if f in input_df.columns]
                
                top_features = [(available_features[i], round(importances[i]*100, 2)) for i in indices if i < len(available_features)]
                
                importances_df = pd.DataFrame({
                    'Característica': [f"{f} ({feature_descriptions[f]})" for f, _ in top_features[:5]],
                    'Importancia (%)': [i for _, i in top_features[:5]]
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x='Importancia (%)', y='Característica', data=importances_df, ax=ax)
                st.pyplot(fig)

with tab3:
    st.header("Predicción por Lotes (CSV)")
    
    if models is None:
        st.warning("No se pudieron cargar los modelos. Por favor, guarda tus modelos primero.")
    else:
        st.write(f"Usando el modelo: **{selected_model}**")
        
        # Información sobre el formato del CSV
        st.info(f"""
        El archivo CSV debe contener columnas con los siguientes nombres: 
        {', '.join(['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'cp', 'restecg'])}
        
        Puedes descargar una plantilla con el formato correcto usando el botón de abajo.
        """)
        
        # Crear una plantilla de CSV para descargar
        template_df = pd.DataFrame({
            'age': [feature_ranges['age'][2]],
            'sex': [feature_ranges['sex'][2]],
            'trestbps': [feature_ranges['trestbps'][2]],
            'chol': [feature_ranges['chol'][2]],
            'fbs': [feature_ranges['fbs'][2]],
            'thalach': [feature_ranges['thalach'][2]],
            'exang': [feature_ranges['exang'][2]],
            'oldpeak': [feature_ranges['oldpeak'][2]],
            'cp': [feature_ranges['cp'][2]],
            'restecg': [feature_ranges['restecg'][2]]
        })
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="Descargar plantilla CSV",
            data=csv_template,
            file_name="plantilla_prediccion.csv",
            mime="text/csv"
        )
        
        # Widget para cargar archivo
        uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
        
        if uploaded_file is not None:
            # Cargar el CSV
            try:
                data = pd.read_csv(uploaded_file)
                
                # Mostrar vista previa
                st.subheader("Vista previa de los datos:")
                st.write(data.head())
                
                # Verificar que las columnas existan
                expected_input_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'cp', 'restecg']
                missing_cols = [col for col in expected_input_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Faltan las siguientes columnas en el CSV: {', '.join(missing_cols)}")
                else:
                    # Botón para procesar
                    if st.button("Procesar Datos"):
                        with st.spinner("Procesando datos..."):
                            # Preparar los datos para la predicción, incluyendo la codificación
                            X = []
                            for _, row in data.iterrows():
                                input_data = row.to_dict()
                                prepared_data = prepare_input_for_model(input_data, models[selected_model])
                                X.append(prepared_data.values[0])  # Extraer el array de valores
                            X = np.array(X)
                            
                            # Hacer predicciones
                            model = models[selected_model]
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X)
                            
                            # Agregar predicciones al DataFrame
                            results = data.copy()
                            results['Predicción'] = predictions
                            results['Probabilidad_Enfermedad'] = probabilities[:, 1]
                            
                            # Mostrar resultados
                            st.subheader("Resultados:")
                            st.write(results)
                            
                            # Estadísticas de las predicciones
                            st.subheader("Distribución de Predicciones:")
                            fig, ax = plt.subplots()
                            sns.countplot(x='Predicción', data=results, ax=ax)
                            ax.set_xticklabels(['Sin enfermedad', 'Con enfermedad'])
                            st.pyplot(fig)
                            
                            # Opción para descargar resultados
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Descargar resultados como CSV",
                                data=csv,
                                file_name="predicciones.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

with tab4:
    st.header("Rendimiento de Modelos")
    
    if models is None:
        st.warning("No se pudieron cargar los modelos. Por favor, guarda tus modelos primero.")
    else:
        st.write("Esta sección te permite comparar el rendimiento de los diferentes modelos.")
        
        # Detalles de los modelos
        st.subheader("Detalles de los Modelos:")
        
        for name, model in models.items():
            with st.expander(f"Modelo: {name}"):
                st.write(f"**Tipo:** {type(model).__name__}")
                
                # Mostrar parámetros
                st.write("**Parámetros:**")
                params = model.get_params()
                filtered_params = {k: v for k, v in params.items() if k in ['n_estimators', 'learning_rate', 'max_depth', 'subsample']}
                params_df = pd.DataFrame([filtered_params])
                st.write(params_df)
                
                # Mostrar importancia de características si están disponibles
                if hasattr(model, 'feature_importances_'):
                    st.write("**Importancia de características:**")
                    importances = model.feature_importances_
                    
                    # Muestra información de depuración
                    st.write(f"Longitud de importances: {len(importances)}")
                    st.write(f"Longitud de feature_names: {len(feature_names)}")
                    
                    # Asegúrate de que indices solo tenga valores válidos
                    # Primero ordena los índices por importancia
                    indices = np.argsort(importances)[::-1]
                    
                    # Luego filtra solo los índices válidos
                    valid_indices = [i for i in indices if i < len(feature_names)]
                    
                    # Crea el DataFrame solo con índices válidos
                    importance_df = pd.DataFrame({
                        'Característica': [feature_names[i] for i in valid_indices],
                        'Descripción': [feature_descriptions.get(feature_names[i], "Sin descripción") for i in valid_indices],
                        'Importancia (%)': [round(importances[i]*100, 2) for i in valid_indices]
                    })
                    
                    st.write(importance_df)
                    
                    # Visualización de importancia
                    if len(importance_df) > 0:  # Solo muestra el gráfico si hay datos
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(x='Importancia (%)', y='Característica', data=importance_df, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("No hay características con importancia para mostrar")
        
        # Métricas de rendimiento (ejemplo)
        st.subheader("Métricas de Rendimiento:")
        st.write("Para mostrar las métricas reales de tus modelos, necesitas guardarlas junto con los modelos.")
        
        # Simulamos algunas métricas para la demo
        metrics = {
            'AdaBoost': {'Accuracy': 0.85, 'F1-Score': 0.83, 'ROC-AUC': 0.90},
            'Gradient Boosting': {'Accuracy': 0.87, 'F1-Score': 0.86, 'ROC-AUC': 0.92},
            'XGBoost': {'Accuracy': 0.89, 'F1-Score': 0.88, 'ROC-AUC': 0.94}
        }
        
        metrics_df = pd.DataFrame(metrics).T
        st.write(metrics_df)
        
        # Visualización de métricas
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_ylim(0.8, 1.0)
        ax.set_title('Comparación de Métricas por Modelo')
        ax.set_ylabel('Puntuación')
        ax.legend(title='Métrica')
        st.pyplot(fig)

# Instrucciones para guardar los modelos
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ ¿Cómo guardar tus modelos?"):
    st.code("""
# Añade este código al final de tu notebook:
import pickle

# Guardar todos los modelos optimizados
with open('best_models.pkl', 'wb') as file:
    pickle.dump(best_models, file)

print("¡Modelos guardados como 'best_models.pkl'!")
    """, language="python")
    
    st.write("Asegúrate de guardar este archivo en la misma carpeta que tu app.py")

# Información sobre el autor
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por: [Gabriel sosa]")


