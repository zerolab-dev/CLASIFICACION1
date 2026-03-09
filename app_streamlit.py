# Debe direccionar VS Code a la carpeta con los archivos:
# 1.- Archivo
# 2.- Abrir carpeta. Debe dar click en la carpeta que contiene los archivos de interés
#3.- A la izquierda, en el explorador deberá poder visualizar todos los archivos
#------------------------------------------------------------------------------------------------

# CÓDIGO STREAMLIT
# Ir a:   Ver/Terminal
# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv'): coloca este código
#   python -m venv venv

#---------------------------------------------------------------------------------------
# Luego de crear el ambiente virtual, lo activas
#   .\venv\Scripts\activate   # En Windows
#---------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Cuando vuelva a iniciar sesión, debe volver a activar el ambiente virtual, ya no lo debe crear.
# En este caso debes abrir la carpeta con los archivos del caso.
#---------------------------------------------------------------------------------------------

# Instala la versión específica de scikit-learn
#   pip install scikit-learn==1.2.2
# Instala otras dependencias, incluyendo Streamlit
#  pip install streamlit pandas joblib
#-------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
#      Get-ExecutionPolicy                           Si es Restricted; ejecuta
#      Set-ExecutionPolicy RemoteSigned              Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate

import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# -------------------------PROCESO DE DESPLIEGUE------------------------------
# En consola:
# pip install scikit-learn==1.3.2

# 01 --------------------------Load the model-------------------------------------------
clf = load('modelo_rfchurn_tunning.joblib')

# 02---------------- Variables globales para los campos del formulario-----------------------
geography_options = ['France', 'Spain', 'Germany']
geography = ''
age = 0
balance = 0.0
num_of_products = 1
is_active_member_options = [0, 1]
is_active_member = 0

# 03 Reseteo------------- Flag to track error---------------------------------------
error_flag = False

# Reset inputs function
def reset_inputs():
    global geography, age, balance, num_of_products, is_active_member, error_flag
    geography = ''
    age = 0
    balance = 0.0
    num_of_products = 1
    is_active_member = 0
    error_flag = False

# Inicializar variables
reset_inputs()
# -----------------------------------------------------------------------------------------------

# ------------------------Título centrado-------------------------------------------------
st.title("Modelo Predictivo de Churn en un Banco con Random Forest Classifier")
st.markdown("Este modelo predice si un cliente cerrará su cuenta en el banco (Churn) en base a diferentes características.")
st.markdown("---")

# ----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs():
    global error_flag
    if any(val < 0 for val in [age, balance]):
        st.error("No se permiten valores negativos. Por favor, ingrese valores válidos en todos los campos.")
        error_flag = True
    else:
        error_flag = False

# ------------------------------------ Formulario en dos columnas------------------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    # Input fields en la primera columna
    with col1:
        geography = st.selectbox("**Geografía**", geography_options)
        age = st.number_input("**Edad**", min_value=0.0, value=float(age), step=1.0)
        balance = st.number_input("**Balance**", min_value=0.0, value=balance, step=1.0,
                                  help="Saldo en la cuenta del cliente")
        num_of_products = st.number_input("**Número de productos**", min_value=1, value=num_of_products, step=1)
        
    # Input fields en la segunda columna
    with col2:
        is_active_member = st.selectbox("**¿Es miembro activo? 0 NO; 1: SÍ**", is_active_member_options)

    # ----------------------------------------- Boton de Predecir-------------------------------------------------
    predict_button = st.form_submit_button("Predecir")

# Validar que no haya valores negativos en los campos cuando se presiona el botón
# Si hay error no permita seguir tipeando!!!!!!!!!!!!!!!!!!!
if predict_button and error_flag:
    st.stop()

if predict_button and not error_flag:
    # Crear DataFrame
    data = {
        'Geography': [geography],
        'Age': [age],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'IsActiveMember': [is_active_member]
    }
    df = pd.DataFrame(data)

    # Realizar predicción
    probabilities_classes = clf.predict_proba(df)[0]

    # Obtener la clase con la mayor probabilidad
    class_predicted = np.argmax(probabilities_classes)

    # Asignar salida y probabilidad según la clase predicha
    # En el script original: #Exited: 0 Cliente retenido;  1 Cliente cerró cuenta
    if class_predicted == 0:
        outcome = "Cliente Retenido"
        probability_churn = probabilities_classes[0]
        style_result = 'background-color: lightgreen; font-size: larger;'
    else:
        outcome = "Churn (Cliente cerró cuenta)"
        probability_churn = probabilities_classes[1]
        style_result = 'background-color: lightcoral; font-size: larger;'

    # Mostrar resultado con estilo personalizado
    result_html = f"<div style='{style_result}'>La predicción fue de clase '{outcome}' con una probabilidad de {round(float(probability_churn), 4)}</div>"
    st.markdown(result_html, unsafe_allow_html=True)

# --------------------------- Boton de Resetear-------------------------------------
if st.button("Resetear"):
    # Resetear inputs
    reset_inputs()

# streamlit run app_streamlit.py       en la consola


#pip freeze > requirements.txt
# genera el archivo requirements para el despliegue web.

