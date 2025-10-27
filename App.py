import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from io import StringIO

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador Olfativo 3D")

# --- 2. DATOS DE EJEMPLO (MOCK DATABASE) ---
@st.cache_data
def get_mock_ingredients():
    """
    Crea un DataFrame de ejemplo para simular la base de datos.
    """
    data = {
        'nombre': [
            "Limoneno", "Linalool", "Vainillina", "Etil Maltol", "Sándalo (Sandalore)", 
            "Iso E Super", "Ambroxan", "Rosa Óxido", "Cis-3-Hexenol", "Eugenol",
            "Indol", "Menta Piperita", "Geraniol", "Citronelol", "Patchouli"
        ],
        'familia_olfativa': [
            "Cítrico", "Floral", "Gourmand", "Gourmand", "Amaderado",
            "Amaderado", "Ámbar", "Floral", "Verde", "Especiado",
            "Animal", "Herbal", "Floral", "Floral", "Amaderado"
        ],
        'logp': [
            4.35, 2.84, 1.21, 1.35, 4.10, 
            5.50, 4.80, 2.70, 1.40, 2.27,
            2.30, 2.80, 3.40, 3.25, 4.50
        ],
        'odor_value': [
            800, 600, 1500, 1200, 300,
            200, 400, 900, 1100, 700,
            50, 850, 750, 650, 250
        ],
        'cost': [
            10, 25, 150, 180, 450,
            60, 800, 300, 90, 40,
            1200, 30, 50, 55, 120
        ]
    }
    df = pd.DataFrame(data)
    
    # Renombrar columnas para compatibilidad con el código original
    df = df.rename(columns={
        'logp': 'logP',
        'familia_olfativa': 'Familia Olfativa'
    })
    return df

# --- 3. LÓGICA DE PROCESAMIENTO (PCA) ---
@st.cache_data
def process_ingredients_pca(df):
    """
    Toma el DataFrame de ingredientes y aplica PCA.
    """
    if df.empty:
        return pd.DataFrame()

    # Asegurarse de que las columnas existen
    features_cols = ['logP', 'odor_value', 'cost']
    if not all(col in df.columns for col in features_cols):
        st.error("El DataFrame de ingredientes no tiene las columnas necesarias (logP, odor_value, cost).")
        return pd.DataFrame()
        
    features = df[features_cols].values
    
    # Aplicar PCA para reducir a 3 dimensiones
    pca = PCA(n_components=3)
    coords = pca.fit_transform(features)
    
    df['Dim 1 (Olfativo)'] = coords[:, 0]
    df['Dim 2 (Tonalidad)'] = coords[:, 1]
    df['Dim 3 (Impacto)'] = coords[:, 2]
    
    return df

# --- 4. INTERFAZ DE USUARIO (STREAMLIT) ---

st.title("🧪 Simulador Olfativo Profesional 3D")
st.info("Esta es una demo con datos de ejemplo. La funcionalidad de IA y la base de datos han sido deshabilitadas.")

# Cargar y procesar datos
df_ingredients = get_mock_ingredients()
df_processed = process_ingredients_pca(df_ingredients)

if df_processed.empty:
    st.error("No se pudieron cargar o procesar los datos de ingredientes.")
else:
    # Inicializar estado de sesión para la fórmula
    if 'formula' not in st.session_state:
        st.session_state.formula = {} # Un diccionario para guardar {nombre: cantidad}

    # Definir pestañas
    tab1, tab2, tab3 = st.tabs(["Visualizador 3D", "Explorador de Datos", "Diseñador de Fórmulas"])

    # --- Pestaña 1: Visualizador 3D ---
    with tab1:
        st.header("Mapa Olfativo 3D (Análisis PCA)")
        st.markdown("Visualiza la relación entre ingredientes basado en sus propiedades (costo, logP, valor olfativo).")
        
        fig = px.scatter_3d(
            df_processed,
            x='Dim 1 (Olfativo)',
            y='Dim 2 (Tonalidad)',
            z='Dim 3 (Impacto)',
            color='Familia Olfativa',
            hover_name='nombre',
            hover_data=['logP', 'cost', 'odor_value'],
            height=700
        )
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title='Dim 1 (Olfativo)',
                yaxis_title='Dim 2 (Tonalidad)',
                zaxis_title='Dim 3 (Impacto)'
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Pestaña 2: Explorador de Datos ---
    with tab2:
        st.header("Base de Datos de Ingredientes")
        
        # Filtros
        familias_unicas = df_processed['Familia Olfativa'].unique()
        familias_seleccionadas = st.multiselect(
            "Filtrar por familia olfativa:",
            options=familias_unicas,
            default=familias_unicas
        )
        
        df_filtrado = df_processed[df_processed['Familia Olfativa'].isin(familias_seleccionadas)]
        
        st.dataframe(df_filtrado)

    # --- Pestaña 3: Diseñador de Fórmulas ---
    with tab3:
        st.header("Diseñador de Fórmulas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Añadir Ingrediente")
            
            # Formulario para añadir ingredientes
            with st.form(key="add_ingredient_form", clear_on_submit=True):
                ingrediente_sel = st.selectbox(
                    "Selecciona un ingrediente:",
                    options=df_processed['nombre']
                )
                cantidad = st.number_input(
                    "Cantidad (partes):",
                    min_value=0.1,
                    value=1.0,
                    step=0.5
                )
                submit_button = st.form_submit_button(label="Añadir a la Fórmula")
                
                if submit_button:
                    if ingrediente_sel in st.session_state.formula:
                        st.session_state.formula[ingrediente_sel] += cantidad
                    else:
                        st.session_state.formula[ingrediente_sel] = cantidad
                    st.success(f"Añadido {cantidad} de {ingrediente_sel}")

        with col2:
            st.subheader("Fórmula Actual")
            
            if not st.session_state.formula:
                st.info("Tu fórmula está vacía. Añade ingredientes desde el panel izquierdo.")
            else:
                formula_str = "FÓRMULA DE PERFUME\n" + "="*20 + "\n"
                total_partes = 0
                
                # Mostrar fórmula como un DataFrame
                formula_df_data = []
                for nombre, cantidad in st.session_state.formula.items():
                    formula_df_data.append({"Ingrediente": nombre, "Cantidad": cantidad})
                    formula_str += f"{nombre}: {cantidad}\n"
                    total_partes += cantidad
                
                formula_df = pd.DataFrame(formula_df_data)
                st.dataframe(formula_df)
                
                st.metric(label="Total de Partes", value=f"{total_partes:.2f}")
                
                formula_str += "="*20 + "\n"
                formula_str += f"TOTAL: {total_partes:.2f} partes\n"
                
                # Botón para limpiar
                if st.button("Limpiar Fórmula", type="primary"):
                    st.session_state.formula = {}
                    st.rerun()
                
                # Botón de descarga (reemplaza al PDF)
                st.download_button(
                    label="Descargar Fórmula (.txt)",
                    data=StringIO(formula_str).getvalue(),
                    file_name="formula_perfume.txt",
                    mime="text/plain"
                )


