
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="SSDA - Guía Agrofinanciera y Tipología Municipal",
    page_icon="🌾",
    layout="wide"
)

def fmt_miles(valor, decimales=0):
    if decimales == 0:
        return f"{round(valor):,}".replace(",", ".")
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")

# Cargar la Tabla Maestra Definitiva (con K=5 y submuestra continental)
@st.cache_data
def cargar_datos():
    ruta = "data/tabla_maestra_con_5_clusters_definitivo.csv"
    df = pd.read_csv(ruta, dtype={'DIVIPOLA': str}, low_memory=False)

    mapeo_clusters = {
        0: "C0. Transición y Ladera Intermedia",
        1: "C1. Empresarial de Altura y Alta Tecnología",
        2: "C2. Economía Campesina y Suelos Ácidos",
        3: "C3. Nodo Financiero y Urbano (Medellín)",
        4: "C4. Enclave de Agricultura Intensiva y Alta Fertilización"
    }

    col_cluster = next((c for c in ['CLUSTER_KMEANS', 'CLUSTER', 'cluster'] if c in df.columns), None)
    if col_cluster is not None:
        df['CLUSTER'] = df[col_cluster]
        if 'NOMBRE_PERFIL' not in df.columns:
            df['NOMBRE_PERFIL'] = df[col_cluster].map(mapeo_clusters).fillna("Perfil General")
    elif 'NOMBRE_PERFIL' not in df.columns:
        df['CLUSTER'] = 0
        df['NOMBRE_PERFIL'] = "Perfil General"

    return df
df = cargar_datos()

# ENCABEZADO PRINCIPAL
st.title("🌾 Sistema de Soporte a la Decisión Agrofinanciera (SSDA)")
st.markdown("### *Herramienta de diagnóstico territorial, enmiendas edáficas y caracterización municipal (Modelo K-Means K=5)*")
st.divider()

# BARRA LATERAL - SELECCIÓN JERÁRQUICA DE UBICACIÓN
st.sidebar.header("📍 Parámetros del Productor")

# Detectar nombres reales de columnas de departamento y municipio
col_dpto = next((c for c in ['DEPARTAMENTO', 'Departamento', 'dpto'] if c in df.columns), 'DEPARTAMENTO')
col_mun = next((c for c in ['MUNICIPIO', 'Municipio', 'nombre_municipio'] if c in df.columns), 'MUNICIPIO')

departamentos_disponibles = sorted(df[col_dpto].dropna().unique())
dep_sel = st.sidebar.selectbox("Seleccione su Departamento:", departamentos_disponibles)

municipios_disponibles = sorted(df[df[col_dpto] == dep_sel][col_mun].dropna().unique())
mun_sel = st.sidebar.selectbox("Seleccione su Municipio:", municipios_disponibles)

datos_mun = df[(df[col_dpto] == dep_sel) & (df[col_mun] == mun_sel)].iloc[0]

# Catálogo técnico de requerimientos de pH
requerimientos_cultivos = {
    'Café': {'PH_MIN': 5.0, 'PH_MAX': 6.5, 'FUENTE': 'Cenicafé / UPRA - Criterios de Aptitud'},
    'Plátano': {'PH_MIN': 5.5, 'PH_MAX': 7.0, 'FUENTE': 'UPRA / AGROSAVIA - Manual de Suelos'},
    'Maíz': {'PH_MIN': 5.5, 'PH_MAX': 7.5, 'FUENTE': 'UPRA - Zonificación de Aptitud'},
    'Cacao': {'PH_MIN': 6.0, 'PH_MAX': 7.5, 'FUENTE': 'AGROSAVIA - Guía Técnica'},
    'Aguacate Hass': {'PH_MIN': 5.5, 'PH_MAX': 6.5, 'FUENTE': 'UPRA - Guía de Frutales'},
    'Papa': {'PH_MIN': 4.5, 'PH_MAX': 5.5, 'FUENTE': 'AGROSAVIA - Sistemas de Producción de Altura'},
    'Zanahoria': {'PH_MIN': 6.0, 'PH_MAX': 7.0, 'FUENTE': 'UPRA - Manual Hortícola'},
    'Arroz': {'PH_MIN': 5.0, 'PH_MAX': 6.5, 'FUENTE': 'FEDARROZ / UPRA - Zonificación Cerealera'},
    'Caña Panelera': {'PH_MIN': 5.5, 'PH_MAX': 7.5, 'FUENTE': 'AGROSAVIA - Cultivo de Caña Panelera'},
    'Palma de Aceite': {'PH_MIN': 4.0, 'PH_MAX': 6.0, 'FUENTE': 'FEDEPALMA / UPRA - Suelos para Oleaginosas'}
}

cultivo_sel = st.sidebar.selectbox("Seleccione el Cultivo:", list(requerimientos_cultivos.keys()))

st.sidebar.divider()
st.sidebar.markdown("### 🗃️ Metadatos del Modelo")
st.sidebar.metric(label="Municipios Continentales Evaluados", value=fmt_miles(df.shape[0], 0))
st.sidebar.metric(label="Variables Vectoriales (Modelo)", value="8")

perfil_nombre = str(datos_mun.get('NOMBRE_PERFIL', 'Perfil General'))
cluster_id = int(datos_mun.get('CLUSTER', 0))

st.header(f"🎯 Clasificación Tipológica: {mun_sel} ({dep_sel})")

descripciones_clusters = {
    0: "Transición y ladera intermedia: Ruralidad dispersa con fertilidad moderada y acidez manejable.",
    1: "Empresarial de altura y alta tecnología: Zonas frías con alta saturación de bases y rendimientos destacados.",
    2: "Economía campesina y suelos ácidos: Sistemas con limitaciones edáficas severas y menor disponibilidad nutricional.",
    3: "Nodo financiero y urbano (Medellín): Epicentro atípico de intermediación crediticia y masiva colocación de recursos.",
    4: "Enclave de agricultura intensiva y alta fertilización: Nicho con niveles extremos de fósforo (Bray II > 195 ppm)."
}

st.info(f"📌 **Asignación Algorítmica (K-Means K=5):** **{perfil_nombre}**\n\n"
        f"💡 **Caracterización del Clúster:** {descripciones_clusters.get(cluster_id, 'Perfil territorial estándar.')}")

st.divider()

# 📖 GUÍA DE INTERPRETACIÓN DUAL (PRODUCTOR Y ASESOR DE CRÉDITO)
with st.expander("📖 Guía de Interpretación de Tipologías para Productores y Asesores de Crédito", expanded=False):
    st.markdown("### 🌾 Orientación Estratégica por Clúster Territorial")
    st.markdown("Esta sección traduce las variables analíticas del modelo en recomendaciones prácticas orientadas tanto al manejo de la finca como a la evaluación y colocación de crédito agropecuario.")

    st.markdown("---")
    st.markdown("#### **C0. Transición y Ladera Intermedia**")
    st.markdown("- **Para el Productor:** Zona de ladera con fertilidad moderada y acidez manejable. Se recomienda implementar prácticas de conservación de suelos y fertilización de soporte para proteger la capa arable.")
    st.markdown("- **Para el Asesor de Crédito:** Perfil de riesgo intermedio. Ideal para créditos de fomento agrícola tradicional con planes de amortización ajustados a los ciclos de cosecha de ladera.")

    st.markdown("#### **C1. Empresarial de Altura y Alta Tecnología**")
    st.markdown("- **Para el Productor:** Agricultura intensiva en pisos térmicos fríos (papa, frutales, hortalizas) con alta inversión y excelentes rendimientos. Oportunidad para tecnificación avanzada.")
    st.markdown("- **Para el Asesor de Crédito:** Unidades productivas estables con alta capacidad de pago y bajo riesgo operacional. Foco prioritario para líneas de modernización, maquinaria y capital de trabajo.")

    st.markdown("#### **C2. Economía Campesina y Suelos Ácidos**")
    st.markdown("- **Para el Productor:** Sistemas de economía campesina que enfrentan suelos ácidos y menor disponibilidad nutricional. Prioridad absoluta en enmiendas correctivas con cal agrícola.")
    st.markdown("- **Para el Asesor de Crédito:** Segmento que requiere esquemas de microfinanzas rurales, créditos asociativos y articulación con subsidios o incentivos (como el Incentivo al Seguro Agropecuario - ISA) para mitigar la vulnerabilidad climática y edáfica.")

    st.markdown("#### **C3. Nodo Financiero y Urbano (Medellín)**")
    st.markdown("- **Para el Productor:** Municipio que funge como centro de acopio, comercialización y gestión de servicios agroindustriales y financieros.")
    st.markdown("- **Para el Asesor de Crédito:** *Outlier* estructural de colocación masiva e intermediación financiera. Corresponde a operaciones de banca corporativa, redescuento y grandes centrales de abasto.")

    st.markdown("#### **C4. Enclave de Agricultura Intensiva y Alta Fertilización**")
    st.markdown("- **Para el Productor:** Zonas con saturación extrema de Fósforo (Bray II > 195 ppm) e intensa rotación hortícola. Cuidado con el exceso de nutrientes para evitar pasivos ambientales.")
    st.markdown("- **Para el Asesor de Crédito:** Alta dinámica productiva y comercial. Requiere financiamiento especializado para paquetes tecnológicos de alto valor y seguimiento al manejo ambiental.")

st.divider()

st.header(f"📊 1. Diagnóstico Edáfico y Productivo")

col1, col2, col3 = st.columns(3)

ph_val = datos_mun.get('PH_AGUASUELO', 5.5)
rend_val = datos_mun.get('RENDIMIENTO', 10.0)
mat_org = datos_mun.get('MATERIA_ORGANICA', 0.0)

col1.metric("pH Promedio del Suelo", f"{ph_val:.2f}".replace(".", ","), delta="Crítico (<5,5)" if ph_val < 5.5 else "Adecuado", delta_color="inverse" if ph_val < 5.5 else "normal")
col2.metric("Rendimiento Promedio", f"{fmt_miles(rend_val, 2)} t/ha")
col3.metric("Materia Orgánica (%)", f"{fmt_miles(mat_org, 2)}%")

st.divider()

# EVALUACIÓN DE APTITUD POR PH
st.subheader(f"🌱 Evaluación de Aptitud por pH para el Cultivo: {cultivo_sel}")
info_cultivo = requerimientos_cultivos[cultivo_sel]
ph_min_c = info_cultivo['PH_MIN']
ph_max_c = info_cultivo['PH_MAX']
fuente_c = info_cultivo['FUENTE']

c_apt1, c_apt2 = st.columns([2, 1])

with c_apt1:
    # ---> AQUÍ REEMPLAZAS EL BLOQUE if / elif / else <---
    ph_str = f"{ph_val:.2f}".replace(".", ",")
    if ph_min_c <= ph_val <= ph_max_c:
        st.success(f"✅ **Apto por pH**: El pH promedio del suelo ({ph_str}) se encuentra dentro del rango óptimo técnico para el **{cultivo_sel}** ([{ph_min_c} - {ph_max_c}]).")
    elif ph_val < ph_min_c:
        st.warning(f"⚠️ **Restricción por Acidez**: El pH del suelo ({ph_str}) es inferior al mínimo requerido para el {cultivo_sel} ({ph_min_c}). Se requiere enmienda correctiva con cal.")
    else:
        st.warning(f"⚠️ **Restricción por Alcalinidad**: El pH del suelo ({ph_str}) supera el máximo recomendado para el {cultivo_sel} ({ph_max_c}).")

with c_apt2:
    st.info(f"📚 **Respaldo Técnico**:\n{fuente_c}")

st.divider()

st.header("💡 2. Recomendador y Calculadora de Enmienda (Cal Agrícola)")
st.markdown("Guía orientativa para corregir acidez edáfica y desbloquear fósforo en el predio.")

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("⚙️ Parámetros de la Finca")

    # Selector de unidad de área para mayor dinamismo del productor
    unidad_area = st.radio("Seleccione la unidad para ingresar el área:", ["Hectáreas (Ha)", "Metros Cuadrados (m²)"], horizontal=True)

    if unidad_area == "Hectáreas (Ha)":
        area_input = st.number_input("Número de hectáreas a corregir (Ha):", min_value=0.1, value=2.0, step=0.5)
        hectareas_totales = area_input
    else:
        area_input = st.number_input("Área en metros cuadrados a corregir (m²):", min_value=100.0, value=10000.0, step=500.0)
        hectareas_totales = area_input / 10000.0  # Conversión a Hectáreas

    ph_finca = st.slider("pH actual estimado en terreno:", min_value=4.0, max_value=7.0, value=float(np.round(ph_val, 1)), step=0.1)
    precio_ton_cal = st.number_input("Costo estimado de la Tonelada de Cal (COP):", min_value=100000, value=350000, step=25000)

    dosis_ton_ha = (5.5 - ph_finca) * 1.5 if ph_finca < 5.5 else 0.0
    total_ton = dosis_ton_ha * hectareas_totales
    total_kg = total_ton * 1000  # Conversión a Kilogramos
    costo_total = total_ton * precio_ton_cal

with col_sim2:
    st.subheader("📋 Recomendación de Manejo")
    if dosis_ton_ha > 0:
        dosis_kg_ha = dosis_ton_ha * 1000
        str_dosis_ton = fmt_miles(dosis_ton_ha, 0)
        str_dosis_kg = fmt_miles(dosis_kg_ha, 0)
        str_total_ton = fmt_miles(total_ton, 0)
        str_total_kg = fmt_miles(total_kg, 0)
        str_area_m2 = fmt_miles(hectareas_totales * 10000, 0)
        str_costo = fmt_miles(costo_total, 0)
        str_hectareas = fmt_miles(hectareas_totales, 2)

        st.warning(f"⚠️ **Suelo Ácido Detectado:** Se recomienda aplicar **{str_dosis_kg} kg/Ha** ({str_dosis_ton} Toneladas/Ha) de cal agrícola.")
        st.info(f"📦 **Requerimiento Total ({str_hectareas} Ha / {str_area_m2} m²):** **{str_total_ton} Toneladas** (**{str_total_kg} Kilogramos**)")
        st.success(f"💰 **Inversión Estimada en Enmienda:** **${str_costo} COP**")
    else:
        st.success("✅ **Suelo en Rango Óptimo:** Su nivel de pH no requiere encalado correctivo inmediato.")

st.divider()

st.header("📈 3. Posicionamiento Nacional (Suelos vs. Fomento Crediticio)")

df['CREDITO_MILLONES'] = df['COLOCACION_PROMEDIO_ANUAL'] / 1e6

colores_clusters = {
    "C0. Transición y Ladera Intermedia": "#1f77b4",
    "C1. Empresarial de Altura y Alta Tecnología": "#ff7f0e",
    "C2. Economía Campesina y Suelos Ácidos": "#2ca02c",
    "C3. Nodo Financiero y Urbano (Medellín)": "#d62728",
    "C4. Enclave de Agricultura Intensiva y Alta Fertilización": "#9467bd"
}

fig = px.scatter(
    df,
    x='PH_AGUASUELO',
    y='CREDITO_MILLONES',
    color='NOMBRE_PERFIL',
    color_discrete_map=colores_clusters,
    hover_name=col_mun,
    title="Matriz Espacial de Clústeres Municipales (K-Means K=5)",
    labels={
        'PH_AGUASUELO': 'pH Promedio del Suelo',
        'CREDITO_MILLONES': 'Colocación Promedio Anual (Millones COP)',
        'NOMBRE_PERFIL': 'Tipología Municipal'
    },
    template="plotly_white",
    height=600
)

fig.update_layout(
    legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, title=dict(text="**Tipologías (K=5)**")),
    margin=dict(t=50, b=50, l=50, r=250)
)

st.plotly_chart(fig, use_container_width=True)
