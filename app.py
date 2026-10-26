import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import requests
from sklearn.cluster import KMeans

# Configuración de la página
st.set_page_config(
    page_title="SSDA - Guía Agrofinanciera",
    page_icon="🌾",
    layout="wide"
)

# Cargar la Tabla Maestra desde Google Drive y asignar los 4 Clústeres del modelo
@st.cache_data
def cargar_datos():
    file_id = "d/1KEIwpS5d-_5fUAGebS-ofhn-yQmuwITw" # <--- Asegúrate de tener aquí tu ID de Drive
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        try:
            df = pd.read_excel(io.BytesIO(response.content))
        except:
            df = pd.read_csv(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error al conectar con Google Drive: {e}")
        df = pd.DataFrame()
        
    if not df.empty:
        # Calcular los billones de crédito
        if 'TOTAL_VALOR_CREDITO' in df.columns:
            df['CREDITO_BILLONES'] = df['TOTAL_VALOR_CREDITO'] / 1e12
        
        # Aplicar el modelo de K-Means (k=4) si no viene precalculado
        if 'NOMBRE_PERFIL' not in df.columns:
            features = df[['PCT_SUELOS_ACIDOS', 'CREDITO_BILLONES']].fillna(0)
            kmeans = KMeans(n_clusters=4, random_state=42).fit(features)
            
            mapping_perfiles = {
                0: "Perfil 1: Zonas de Transición y Desarrollo Moderado",
                1: "Perfil 2: Alta Absorción Crediticia y Productividad",
                2: "Perfil 3: Alta Vulnerabilidad Edáfica (Suelos Ácidos)",
                3: "Perfil 4: Caso Atípico / Alta Concentración Nacional"
            }
            df['CLUSTER_K4'] = kmeans.labels_
            df['NOMBRE_PERFIL'] = df['CLUSTER_K4'].map(mapping_perfiles)
            
    return df

df = cargar_datos()

# ENCABEZADO PRINCIPAL
st.title("🌾 Sistema de Soporte a la Decisión Agrofinanciera (SSDA)")
st.markdown("### *Herramienta de diagnóstico territorial, enmiendas edáficas y riesgo crediticio para el productor*")
st.divider()

# BARRA LATERAL - SELECCIÓN DE UBICACIÓN Y CULTIVO
st.sidebar.header("📍 Parámetros del Productor")

if not df.empty and 'DEPARTAMENTO_NORM' in df.columns:
    departamentos_disponibles = sorted(df['DEPARTAMENTO_NORM'].dropna().unique())
else:
    departamentos_disponibles = ["BOGOTÁ, D.C."]

dep_sel = st.sidebar.selectbox("Seleccione su Departamento:", departamentos_disponibles)

requerimientos_cultivos = {
    'Café': {'PH_MIN': 5.0, 'PH_MAX': 6.5, 'FUENTE': 'UPRA - Criterios de Aptitud Agropecuaria'},
    'Plátano': {'PH_MIN': 5.5, 'PH_MAX': 7.0, 'FUENTE': 'UPRA / AGROSAVIA - Manual de Suelos'},
    'Maíz': {'PH_MIN': 5.5, 'PH_MAX': 7.5, 'FUENTE': 'UPRA - Zonificación de Aptitud'},
    'Cacao': {'PH_MIN': 6.0, 'PH_MAX': 7.5, 'FUENTE': 'AGROSAVIA - Guía Técnica'}
}

cultivo_sel = st.sidebar.selectbox("Seleccione el Cultivo (EVAS):", list(requerimientos_cultivos.keys()))

if not df.empty and 'DEPARTAMENTO_NORM' in df.columns:
    datos_dep_filtrados = df[df['DEPARTAMENTO_NORM'] == dep_sel]
    datos_dep = datos_dep_filtrados.iloc[0] if not datos_dep_filtrados.empty else pd.Series()
else:
    datos_dep = pd.Series()

# MÓDULO 1: DIAGNÓSTICO
st.header(f"📊 1. Diagnóstico Edáfico y Productivo: {dep_sel}")
col1, col2, col3, col4 = st.columns(4)

ph_val = datos_dep.get('PH_PROMEDIO', 5.5)
pct_acidos = datos_dep.get('PCT_SUELOS_ACIDOS', 50.0)
fosforo_val = datos_dep.get('FOSFORO_MEDIANA', 10.0)
perfil_nombre = str(datos_dep.get('NOMBRE_PERFIL', 'Perfil General'))

col1.metric("pH Promedio del Suelo", f"{ph_val:.2f}", delta="Crítico (<5.5)" if ph_val < 5.5 else "Adecuado", delta_color="inverse" if ph_val < 5.5 else "normal")
col2.metric("% Suelos Ácidos (<5.5)", f"{pct_acidos:.1f}%")
col3.metric("Fósforo Disponible (P)", f"{fosforo_val:.1f} ppm")
col4.metric("Clasificación (Perfil)", perfil_nombre.split(':')[0])

st.divider()

# EVALUACIÓN DE APTITUD
st.subheader(f"🌱 Evaluación de Aptitud para el Cultivo de: {cultivo_sel}")
info_cultivo = requerimientos_cultivos[cultivo_sel]
ph_min_c = info_cultivo['PH_MIN']
ph_max_c = info_cultivo['PH_MAX']
fuente_c = info_cultivo['FUENTE']

c_apt1, c_apt2 = st.columns([2, 1])
with c_apt1:
    if ph_min_c <= ph_val <= ph_max_c:
        st.success(f"✅ **Apto por pH**: El pH promedio ({ph_val:.2f}) está en el rango óptimo para el {cultivo_sel} ({ph_min_c} - {ph_max_c}).")
    elif ph_val < ph_min_c:
        st.warning(f"⚠️ **Restricción por Acidez**: El pH ({ph_val:.2f}) es inferior al mínimo para el {cultivo_sel} ({ph_min_c}).")
    else:
        st.warning(f"⚠️ **Restricción por Alcalinidad**: El pH ({ph_val:.2f}) supera el máximo recomendado.")
with c_apt2:
    st.info(f"📚 **Respaldo Técnico**: {fuente_c}")

st.divider()

# EXPLICACIÓN DE LOS 4 PERFILES
st.markdown("### 🧩 Definición de las 4 Tipologías Territoriales (Perfiles Agrofinancieros)")
with st.expander("📖 Haz clic aquí para ver la explicación detallada de los 4 Perfiles", expanded=False):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("🟢 **Perfil 1: Zonas de Transición y Desarrollo Moderado**")
        st.markdown("Agrupa regiones con niveles moderados de acidez y estabilidad productiva.")
    with p2:
        st.markdown("🔵 **Perfil 2: Alta Absorción Crediticia y Productividad**")
        st.markdown("Identifica zonas con alta acidez edáfica pero rendimiento destacado.")
    with p3:
        st.markdown("🟠 **Perfil 3: Alta Vulnerabilidad Edáfica (Suelos Ácidos)**")
        st.markdown("Concentra departamentos con fuerte dinámica de inversión y retos por acidez.")
    with p4:
        st.markdown("🔴 **Perfil 4: Caso Atípico / Alta Concentración Nacional**")
        st.markdown("Representado por el Valle del Cauca por su alta colocación de crédito.")

st.divider()

# MÓDULO 2: CALCULADORA DE ENMIENDAS
st.header("💡 2. Recomendador y Calculadora de Enmienda (Cal Agrícola)")
col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    hectareas = st.number_input("Número de hectáreas a corregir (Ha):", min_value=0.5, value=2.0, step=0.5)
    ph_finca = st.slider("pH actual estimado del suelo:", min_value=4.0, max_value=7.0, value=float(np.round(ph_val, 1)), step=0.1)
    precio_ton_cal = st.number_input("Costo estimado de la Tonelada de Cal (COP):", min_value=100000, value=350000, step=25000)

    dosis_ton_ha = (5.5 - ph_finca) * 1.5 if ph_finca < 5.5 else 0.0
    total_ton = dosis_ton_ha * hectareas
    costo_total = total_ton * precio_ton_cal

with col_sim2:
    if dosis_ton_ha > 0:
        st.warning(f"⚠️ **Suelo Ácido**: Se recomienda aplicar **{dosis_ton_ha:.2f} Ton/Ha** de cal agrícola.")
        st.info(f"📦 **Total Requerido ({hectareas} Ha)**: **{total_ton:.2f} Toneladas**")
        st.success(f"💰 **Inversión Estimada**: **${costo_total:,.0f} COP**")
    else:
        st.success("✅ **Suelo Óptimo**: No requiere encalado inmediato.")

st.divider()

# MÓDULO 3: MATRIZ NACIONAL E INTERACTIVA
st.header("📈 3. Posicionamiento Nacional (Suelos vs. Crédito FINAGRO)")

if not df.empty and 'CREDITO_BILLONES' in df.columns:
    fig = px.scatter(
        df,
        x='PCT_SUELOS_ACIDOS',
        y='CREDITO_BILLONES',
        color='NOMBRE_PERFIL',
        hover_name='DEPARTAMENTO_NORM',
        size='TOTAL_PERDIDA_AREA' if 'TOTAL_PERDIDA_AREA' in df.columns else None,
        log_y=True,
        title="Matriz Nacional: % Acidez vs. Valor Total Crédito FINAGRO (k=4)",
        labels={
            'PCT_SUELOS_ACIDOS': '% Suelos Ácidos (< 5.5 pH)',
            'CREDITO_BILLONES': 'Monto Crédito (Billones de COP)',
            'NOMBRE_PERFIL': 'Tipología Territorial'
        },
        template="plotly_white",
        height=650
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.85))
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, title="**Perfiles / Tipologías**"),
        margin=dict(t=50, b=50, l=50, r=180)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nota: El tamaño del punto representa el área total perdida (Ha) según registros EVAS.")
else:
    st.info("Cargue los datos de la matriz nacional para visualizar el gráfico.")
