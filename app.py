import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="SSDA - Guía Agrofinanciera y Tipología Municipal",
    page_icon="🌾",
    layout="wide"
)

def fmt_miles(valor, decimales=0):
    if valor is None or pd.isna(valor):
        return "N/D"
    if decimales == 0:
        return f"{round(float(valor)):,}".replace(",", ".")
    texto = f"{float(valor):,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")

MAPEO_CLUSTERS = {
    0: "C0. Transición y Ladera Intermedia",
    1: "C1. Empresarial de Altura y Alta Tecnología",
    2: "C2. Economía Campesina y Suelos Ácidos",
    3: "C3. Nodo Financiero y Urbano (Medellín)",
    4: "C4. Enclave de Agricultura Intensiva y Alta Fertilización"
}

COLORES_CLUSTERS = {
    "C0. Transición y Ladera Intermedia": "#1f77b4",
    "C1. Empresarial de Altura y Alta Tecnología": "#ff7f0e",
    "C2. Economía Campesina y Suelos Ácidos": "#2ca02c",
    "C3. Nodo Financiero y Urbano (Medellín)": "#d62728",
    "C4. Enclave de Agricultura Intensiva y Alta Fertilización": "#9467bd"
}

DESCRIPCIONES_CLUSTERS = {
    0: "Transición y ladera intermedia: Ruralidad dispersa con fertilidad moderada y acidez manejable.",
    1: "Empresarial de altura y alta tecnología: Zonas frías con alta saturación de bases y rendimientos destacados.",
    2: "Economía campesina y suelos ácidos: Sistemas con limitaciones edáficas severas y menor disponibilidad nutricional.",
    3: "Nodo financiero y urbano (Medellín): Epicentro atípico de intermediación crediticia y masiva colocación de recursos.",
    4: "Enclave de agricultura intensiva y alta fertilización: Nicho con niveles extremos de fósforo (Bray II > 195 ppm)."
}

REQUERIMIENTOS_CULTIVOS = {
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

@st.cache_data(show_spinner="Cargando base municipal...")
def cargar_datos():
    ruta = "data/tabla_maestra_con_5_clusters_definitivo.csv"
    df = pd.read_csv(ruta, dtype={'DIVIPOLA': str}, low_memory=False)

    col_cluster = next((c for c in ['CLUSTER_KMEANS', 'CLUSTER', 'cluster'] if c in df.columns), None)
    if col_cluster is not None:
        df['CLUSTER'] = pd.to_numeric(df[col_cluster], errors='coerce').fillna(0).astype(int)
        if 'NOMBRE_PERFIL' not in df.columns:
            df['NOMBRE_PERFIL'] = df['CLUSTER'].map(MAPEO_CLUSTERS).fillna("Perfil General")
    elif 'NOMBRE_PERFIL' not in df.columns:
        df['CLUSTER'] = 0
        df['NOMBRE_PERFIL'] = "Perfil General"

    for col_num in [
        'PH_AGUASUELO', 'RENDIMIENTO', 'MATERIA_ORGANICA',
        'RELACION_CA_MG', 'MAGNESIO_INTERCAMBIABLE', 'COLOCACION_PROMEDIO_ANUAL'
    ]:
        if col_num in df.columns:
            df[col_num] = pd.to_numeric(df[col_num], errors='coerce')

    if 'COLOCACION_PROMEDIO_ANUAL' in df.columns:
        df['CREDITO_MILLONES'] = df['COLOCACION_PROMEDIO_ANUAL'] / 1e6
    else:
        df['CREDITO_MILLONES'] = 0.0

    return df

@st.cache_data(show_spinner=False)
def construir_grafico_posicionamiento(df_plot, col_mun):
    fig = px.scatter(
        df_plot,
        x='PH_AGUASUELO',
        y='CREDITO_MILLONES',
        color='NOMBRE_PERFIL',
        color_discrete_map=COLORES_CLUSTERS,
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
    return fig

def recomendar_tipo_cal(rel_ca_mg, mg_int):
    """Sugiere Cal Dolomítica o Cal Agrícola según relación Ca:Mg y Mg intercambiable."""
    ca_mg_alto = pd.notna(rel_ca_mg) and float(rel_ca_mg) > 4.0
    mg_bajo = pd.notna(mg_int) and float(mg_int) < 1.0
    if ca_mg_alto or mg_bajo:
        motivos = []
        if ca_mg_alto:
            motivos.append(f"relación Ca:Mg elevada ({fmt_miles(rel_ca_mg, 2)})")
        if mg_bajo:
            motivos.append(f"magnesio intercambiable bajo ({fmt_miles(mg_int, 2)} cmol(+)/kg)")
        return "Cal Dolomítica", " y ".join(motivos)
    return "Cal Agrícola", "disponibilidad de magnesio y relación Ca:Mg dentro de parámetros moderados/adecuados"

df = cargar_datos()

# ENCABEZADO PRINCIPAL
st.title("🌾 Sistema de Soporte a la Decisión Agrofinanciera (SSDA)")
st.markdown("### *Herramienta de diagnóstico territorial, enmiendas edáficas y caracterización municipal (Modelo K-Means K=5)*")
st.divider()

# BARRA LATERAL - SELECCIÓN JERÁRQUICA DE UBICACIÓN
st.sidebar.header("📍 Parámetros del Productor")

col_dpto = next((c for c in ['DEPARTAMENTO', 'Departamento', 'dpto'] if c in df.columns), 'DEPARTAMENTO')
col_mun = next((c for c in ['MUNICIPIO', 'Municipio', 'nombre_municipio'] if c in df.columns), 'MUNICIPIO')

departamentos_disponibles = sorted(df[col_dpto].dropna().unique())
dep_sel = st.sidebar.selectbox("Seleccione su Departamento:", departamentos_disponibles, key="sb_dpto")

municipios_disponibles = sorted(df[df[col_dpto] == dep_sel][col_mun].dropna().unique())
mun_sel = st.sidebar.selectbox("Seleccione su Municipio:", municipios_disponibles, key="sb_mun")

cultivo_sel = st.sidebar.selectbox("Seleccione el Cultivo:", list(REQUERIMIENTOS_CULTIVOS.keys()), key="sb_cultivo")

st.sidebar.divider()
st.sidebar.markdown("### 🗃️ Metadatos del Modelo")
st.sidebar.metric(label="Municipios Continentales Evaluados", value=fmt_miles(df.shape[0], 0))
st.sidebar.metric(label="Variables Vectoriales (Modelo)", value="8")

datos_mun = df[(df[col_dpto] == dep_sel) & (df[col_mun] == mun_sel)].iloc[0]
perfil_nombre = str(datos_mun.get('NOMBRE_PERFIL', 'Perfil General'))
cluster_id = int(datos_mun.get('CLUSTER', 0))

st.header(f"🎯 Clasificación Tipológica: {mun_sel} ({dep_sel})")
st.info(
    f"📌 **Asignación Algorítmica (K-Means K=5):** **{perfil_nombre}**\n\n"
    f"💡 **Caracterización del Clúster:** {DESCRIPCIONES_CLUSTERS.get(cluster_id, 'Perfil territorial estándar.')}"
)

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
    st.markdown("- **Para el Productor:** Sistemas de economía campesina que enfrentan suelos ácidos y menor disponibilidad nutricional. Prioridad absoluta en enmiendas correctivas con cal dolomítica o cal agrícola según el balance de bases.")
    st.markdown("- **Para el Asesor de Crédito:** Segmento que requiere esquemas de microfinanzas rurales, créditos asociativos y articulación con subsidios o incentivos (como el Incentivo al Seguro Agropecuario - ISA) para mitigar la vulnerabilidad climática y edáfica.")

    st.markdown("#### **C3. Nodo Financiero y Urbano (Medellín)**")
    st.markdown("- **Para el Productor:** Municipio que funge como centro de acopio, comercialización y gestión de servicios agroindustriales y financieros.")
    st.markdown("- **Para el Asesor de Crédito:** *Outlier* estructural de colocación masiva e intermediación financiera. Corresponde a operaciones de banca corporativa, redescuento y grandes centrales de abasto.")

    st.markdown("#### **C4. Enclave de Agricultura Intensiva y Alta Fertilización**")
    st.markdown("- **Para el Productor:** Zonas con saturación extrema de Fósforo (Bray II > 195 ppm) e intensa rotación hortícola. Cuidado con el exceso de nutrientes para evitar pasivos ambientales.")
    st.markdown("- **Para el Asesor de Crédito:** Alta dinámica productiva y comercial. Requiere financiamiento especializado para paquetes tecnológicos de alto valor y seguimiento al manejo ambiental.")

st.divider()

# 1. DIAGNÓSTICO EDÁFICO Y PRODUCTIVO
st.header("📊 1. Diagnóstico Edáfico y Productivo")

ph_val = float(datos_mun['PH_AGUASUELO']) if pd.notna(datos_mun.get('PH_AGUASUELO')) else 5.5
rend_val = float(datos_mun['RENDIMIENTO']) if pd.notna(datos_mun.get('RENDIMIENTO')) else 0.0
mat_org = float(datos_mun['MATERIA_ORGANICA']) if pd.notna(datos_mun.get('MATERIA_ORGANICA')) else 0.0
rel_ca_mg = datos_mun.get('RELACION_CA_MG', np.nan)
mg_int = datos_mun.get('MAGNESIO_INTERCAMBIABLE', np.nan)

col1, col2, col3 = st.columns(3)
col1.metric(
    "pH Promedio del Suelo",
    f"{ph_val:.2f}".replace(".", ","),
    delta="Crítico (<5,5)" if ph_val < 5.5 else "Adecuado",
    delta_color="inverse" if ph_val < 5.5 else "normal"
)
col2.metric("Rendimiento Promedio", f"{fmt_miles(rend_val, 2)} t/ha")
col3.metric("Materia Orgánica (%)", f"{fmt_miles(mat_org, 2)}%")

st.divider()

# EVALUACIÓN DE APTITUD POR PH
st.subheader(f"🌱 Evaluación de Aptitud por pH para el Cultivo: {cultivo_sel}")
info_cultivo = REQUERIMIENTOS_CULTIVOS[cultivo_sel]
ph_min_c = info_cultivo['PH_MIN']
ph_max_c = info_cultivo['PH_MAX']
fuente_c = info_cultivo['FUENTE']

c_apt1, c_apt2 = st.columns([2, 1])

with c_apt1:
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

# 2. RECOMENDADOR Y CALCULADORA DE ENMIENDA (CAL DOLOMÍTICA / CAL AGRÍCOLA)
@st.fragment
def calculadora_enmienda(ph_mun, rel_ca_mg_mun, mg_int_mun, municipio_nombre):
    st.header("💡 2. Recomendador y Calculadora de Enmienda (Cal Dolomítica / Cal Agrícola)")
    st.markdown("Guía orientativa para corregir acidez edáfica según el perfil del municipio y estimar requerimientos en campo.")

    tipo_sugerido, motivo_tipo = recomendar_tipo_cal(rel_ca_mg_mun, mg_int_mun)

    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        st.subheader("⚙️ Parámetros de la Finca")

        unidad_area = st.radio(
            "Seleccione la unidad para ingresar el área:",
            ["Hectáreas (Ha)", "Metros Cuadrados (m²)"],
            horizontal=True,
            key=f"unidad_{municipio_nombre}"
        )

        if unidad_area == "Hectáreas (Ha)":
            area_input = st.number_input("Número de hectáreas a corregir (Ha):", min_value=0.1, value=2.0, step=0.5, key=f"ha_{municipio_nombre}")
            hectareas_totales = area_input
        else:
            area_input = st.number_input("Área en metros cuadrados a corregir (m²):", min_value=100.0, value=10000.0, step=500.0, key=f"m2_{municipio_nombre}")
            hectareas_totales = area_input / 10000.0

        ph_default = float(np.clip(np.round(ph_mun, 1), 4.0, 7.0))
        ph_finca = st.slider("pH actual estimado en terreno:", min_value=4.0, max_value=7.0, value=ph_default, step=0.1, key=f"ph_{municipio_nombre}")

        idx_tipo = 0 if tipo_sugerido == "Cal Dolomítica" else 1
        tipo_cal_sel = st.selectbox(
            "Tipo de enmienda a calcular:",
            ["Cal Dolomítica", "Cal Agrícola"],
            index=idx_tipo,
            key=f"cal_{municipio_nombre}"
        )

        precio_default = 380000 if tipo_cal_sel == "Cal Dolomítica" else 350000
        precio_ton_cal = st.number_input(
            f"Costo estimado de la Tonelada de {tipo_cal_sel} (COP):",
            min_value=100000,
            value=precio_default,
            step=25000,
            key=f"precio_{municipio_nombre}_{tipo_cal_sel}"
        )

        dosis_ton_ha = (5.5 - ph_finca) * 1.5 if ph_finca < 5.5 else 0.0
        total_ton = dosis_ton_ha * hectareas_totales
        total_kg = total_ton * 1000.0
        costo_total = total_ton * precio_ton_cal

    with col_sim2:
        st.subheader("📋 Recomendación de Manejo")
        st.caption(f"Criterio municipal ({municipio_nombre}): se orienta hacia **{tipo_sugerido}** por {motivo_tipo}.")

        if dosis_ton_ha > 0:
            dosis_kg_ha = dosis_ton_ha * 1000.0
            st.warning(
                f"⚠️ **Suelo Ácido Detectado:** Se recomienda aplicar **{fmt_miles(dosis_kg_ha, 0)} kg/Ha** "
                f"(**{fmt_miles(dosis_ton_ha, 2)} Toneladas/Ha**) de **{tipo_cal_sel}**."
            )
            st.info(
                f"📦 **Requerimiento Total ({fmt_miles(hectareas_totales, 2)} Ha / {fmt_miles(hectareas_totales * 10000, 0)} m²):** "
                f"**{fmt_miles(total_ton, 2)} Toneladas** (**{fmt_miles(total_kg, 0)} Kilogramos**)"
            )
            st.success(f"💰 **Inversión Estimada en Enmienda ({tipo_cal_sel}):** **${fmt_miles(costo_total, 0)} COP**")
        else:
            st.success("✅ **Suelo en Rango Óptimo:** Su nivel de pH no requiere encalado correctivo inmediato.")

calculadora_enmienda(ph_val, rel_ca_mg, mg_int, mun_sel)

st.divider()

# 3. POSICIONAMIENTO NACIONAL
st.header("📈 3. Posicionamiento Nacional (Suelos vs. Fomento Crediticio)")
fig = construir_grafico_posicionamiento(df, col_mun)
st.plotly_chart(fig, use_container_width=True)
