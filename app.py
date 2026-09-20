"""Interfaz Streamlit para clasificar textos en los ODS 1 a 16."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import streamlit as st # libreria para crear la interfaz web de la aplicación
from streamlit_mic_recorder import speech_to_text

# El import registra la función que el pipeline serializado necesita al cargarse.
from src.text_processing import clean_corpus  # noqa: F401
from src.input_processing import (
    MAX_TEXT_CHARACTERS,
    UserInputError,
    extract_document_text,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pipeline_ods.joblib"

ODS_NAMES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
}

ODS_DETAILS = {
    1: "Reducir la pobreza en todas sus formas mediante protección social, acceso a recursos y oportunidades.",
    2: "Poner fin al hambre, mejorar la nutrición y promover sistemas alimentarios y agricultura sostenibles.",
    3: "Garantizar una vida saludable y promover el bienestar de las personas en todas las edades.",
    4: "Asegurar una educación inclusiva y equitativa de calidad y oportunidades de aprendizaje permanente.",
    5: "Alcanzar la igualdad de género y fortalecer los derechos y oportunidades de mujeres y niñas.",
    6: "Garantizar agua segura, saneamiento adecuado y una gestión sostenible de los recursos hídricos.",
    7: "Ampliar el acceso a energía asequible, confiable, sostenible y moderna.",
    8: "Promover crecimiento inclusivo, empleo productivo y condiciones de trabajo decentes.",
    9: "Desarrollar infraestructura resiliente, industrialización sostenible e innovación.",
    10: "Reducir las desigualdades dentro de los países y entre ellos.",
    11: "Construir ciudades y asentamientos inclusivos, seguros, resilientes y sostenibles.",
    12: "Impulsar modalidades sostenibles de producción y consumo y el uso eficiente de los recursos.",
    13: "Adoptar medidas frente al cambio climático, sus impactos y la necesidad de adaptación.",
    14: "Conservar los océanos, los mares y los recursos marinos y utilizarlos de forma sostenible.",
    15: "Proteger los ecosistemas terrestres, los bosques, los suelos y la biodiversidad.",
    16: "Promover sociedades pacíficas, acceso a la justicia e instituciones eficaces e inclusivas.",
}

EXAMPLES = {
    "Educación": "Ampliar el acceso a una educación inclusiva y de calidad exige reducir la brecha digital, formar docentes y garantizar escuelas seguras en las zonas rurales.",
    "Agua": "La comunidad necesita sistemas de agua potable, saneamiento seguro y tratamiento de aguas residuales para prevenir enfermedades y proteger las fuentes hídricas.",
    "Clima": "El plan territorial debe reducir las emisiones, restaurar bosques y preparar a la población ante sequías, inundaciones y otros eventos climáticos extremos.",
}


st.set_page_config(
    page_title="Brújula ODS · Clasificador de textos",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# THESIS: una mesa de análisis que convierte un texto en un dictamen trazable; evita el tablero genérico de métricas.
# OWN-WORLD: papel azul tinta, verde mineral y amarillo de señal; fichas editoriales, reglas finas y numerales tabulares.
# STORY: el usuario aporta evidencia textual, solicita el análisis y comprende el ODS principal, las alternativas y los límites.
# FIRST VIEWPORT: título y método a la izquierda; área de trabajo dominante con entrada y dictamen en dos columnas.
# FORM: mesa de análisis ODS, sexta dirección de la lista; seed ba0e6ab8.
# FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
st.markdown(
    """
    <!-- impeccable-direction: ba0e6ab8 -->
    <style>
    :root {
        --ink: #10263d;
        --paper: #f3f1e8;
        --sheet: #fffdf5;
        --mineral: #13795b;
        --signal: #f4c542;
        --muted: #526474;
        --rule: #b7c0b9;
    }
    html { scroll-behavior: smooth; }
    body, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(243, 241, 232, .92); }
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 1.8rem 2.5rem 4rem; }
    ::selection { background: var(--signal); color: var(--ink); }
    * { scrollbar-color: var(--mineral) var(--paper); }
    h1, h2, h3, label { color: var(--ink) !important; }
    h1 {
        max-width: 780px;
        font-family: "Avenir Next", Avenir, "Trebuchet MS", sans-serif;
        font-size: clamp(2.7rem, 5.2vw, 4.75rem) !important;
        line-height: .93 !important;
        letter-spacing: -.035em !important;
        font-weight: 800 !important;
        margin: 0 0 .8rem !important;
    }
    .lead { max-width: 68ch; font-size: 1.02rem; line-height: 1.55; color: var(--muted); margin-bottom: 1.2rem; }
    .top-rule { height: 7px; width: 112px; background: var(--signal); margin: 0 0 1.2rem; }
    .method {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
        border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink);
        margin: 1.2rem 0 1.6rem;
    }
    .method span { padding: .8rem 1rem; border-right: 1px solid var(--rule); font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; }
    .method span:last-child { border-right: 0; }
    [data-testid="stRadio"] > label { font-weight: 700; color: var(--ink); }
    [data-testid="stFileUploader"] section {
        background: rgba(255,253,245,.65); border-color: var(--rule); border-radius: 12px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--mineral); }
    [data-testid="stTextArea"] textarea {
        min-height: 112px; background: var(--sheet); color: var(--ink);
        border: 1px solid var(--ink); border-radius: 12px; box-shadow: 0 12px 28px rgba(16,38,61,.08);
        caret-color: var(--mineral); font-size: 1rem; line-height: 1.6;
    }
    [data-testid="stTextArea"] textarea:focus { border-color: var(--mineral); box-shadow: 0 0 0 3px rgba(19,121,91,.22); }
    .stButton > button {
        min-height: 48px; border: 0; border-radius: 10px; background: var(--ink); color: white;
        font-weight: 750; padding: .7rem 1.25rem; transition: background .18s ease-out, box-shadow .18s ease-out;
    }
    .stButton > button p { color: white !important; }
    .stButton > button:hover { background: var(--mineral); box-shadow: 0 8px 20px rgba(19,121,91,.2); color: white; }
    .stButton > button:focus-visible { outline: 3px solid var(--signal); outline-offset: 3px; }
    .result {
        min-height: 112px; padding: 1.25rem 1.4rem; border-top: 8px solid var(--signal);
        background: var(--ink); color: white; border-radius: 0 0 14px 14px; box-shadow: 0 14px 30px rgba(16,38,61,.16);
        animation: reveal .42s cubic-bezier(.16,1,.3,1) both;
    }
    @keyframes reveal { from { clip-path: inset(0 0 100% 0); filter: blur(3px); } to { clip-path: inset(0); filter: blur(0); } }
    @media (prefers-reduced-motion: reduce) { .result { animation: none; } html { scroll-behavior: auto; } }
    .result .label { color: #b9d7cb; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
    .result .number { font-size: 4.8rem; line-height: 1; font-weight: 850; font-variant-numeric: tabular-nums; margin: .4rem 0 .25rem; }
    .result .name { color: white; font-size: 1.35rem; font-weight: 700; max-width: 26ch; }
    .empty-result { min-height: 112px; display:flex; align-items:end; padding: 1.25rem; border: 1px solid var(--rule); border-radius: 14px; color: var(--muted); background: rgba(255,253,245,.55); }
    .alt-list { display: grid; gap: 0; margin-top: .35rem; }
    .alt-item { padding: .8rem 0 1rem; border-bottom: 1px solid var(--rule); }
    .alt-item:last-child { border-bottom: 0; }
    .alt-heading { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; font-variant-numeric: tabular-nums; }
    .alt-title { color: var(--ink); font-weight: 750; }
    .alt-margin { color: var(--ink); font-weight: 650; white-space: nowrap; }
    .alt-description { max-width: 72ch; margin: .25rem 0 .6rem; color: var(--muted); font-size: .91rem; line-height: 1.45; }
    .track { height: 8px; background:#dfe4de; border-radius:99px; overflow:hidden; }
    .fill { height:100%; background:var(--mineral); border-radius:99px; }
    .note { padding: 1rem 1.15rem; background:#e1eee8; color:#163c31; border-radius:12px; margin-top:1.6rem; font-size:.92rem; line-height:1.55; }
    .footer-note { color: var(--muted); border-top: 1px solid var(--rule); margin-top: 3.2rem; padding-top: 1.2rem; font-size: .86rem; }
    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] { padding: 1.1rem 1rem 3rem; }
        h1 { font-size: 2.7rem !important; }
        .lead { line-height: 1.45; margin-bottom: .85rem; }
        .top-rule { margin-bottom: .8rem; }
        .method { margin: .9rem 0 1.2rem; }
        .method { grid-template-columns: 1fr 1fr; }
        .method span:nth-child(2) { border-right: 0; }
        .method span:nth-child(-n+2) { border-bottom: 1px solid var(--rule); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando el modelo entrenado…")
def load_model():
    """Carga una sola vez el pipeline reproducible del proyecto."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def alternatives(model, text: str, top_n: int = 3):
    """Devuelve clases y márgenes ordenados; los porcentajes son escala relativa visual."""
    margins = np.asarray(model.decision_function([text])).reshape(-1)
    classes = np.asarray(model.classes_, dtype=int)
    order = np.argsort(margins)[::-1][:top_n]
    selected = margins[order]
    low, high = float(margins.min()), float(margins.max())
    scaled = (selected - low) / (high - low) if high > low else np.ones_like(selected)
    return [(int(classes[i]), float(margins[i]), float(scale)) for i, scale in zip(order, scaled)]


st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
st.title("¿Qué objetivo moviliza este texto?")
st.markdown(
    '<p class="lead">Escribe, adjunta o dicta una idea para explorar políticas, iniciativas y argumentos en español. '
    "El análisis reutiliza el pipeline TF-IDF + LSA + SVM validado en el proyecto.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="method"><span>Escribir · adjuntar · dictar</span><span>TF-IDF</span><span>LSA · 300 dimensiones</span><span>Dictamen ODS</span></div>',
    unsafe_allow_html=True,
)

if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = ""

input_mode = st.radio(
    "Origen del texto",
    ["Escribir", "Adjuntar archivo", "Dictar"],
    horizontal=True,
    help="El contenido extraído o transcrito siempre puede revisarse antes de clasificarlo.",
)

if input_mode == "Escribir":
    example_name = st.selectbox("Probar un texto de ejemplo", ["Escribir mi propio texto", *EXAMPLES.keys()])
    if example_name != st.session_state.get("_last_example"):
        if example_name in EXAMPLES:
            st.session_state.analysis_text = EXAMPLES[example_name]
        st.session_state._last_example = example_name

elif input_mode == "Adjuntar archivo":
    uploaded_document = st.file_uploader(
        "Adjuntar un documento",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=False,
        max_upload_size=5,
        help="Un archivo de hasta 5 MB. Los PDF escaneados sin texto seleccionable requieren OCR.",
    )
    if uploaded_document is not None:
        payload = uploaded_document.getvalue()
        fingerprint = hashlib.sha256(uploaded_document.name.encode("utf-8") + b"\0" + payload).hexdigest()
        if fingerprint != st.session_state.get("_document_fingerprint"):
            st.session_state._document_fingerprint = fingerprint
            try:
                extracted = extract_document_text(uploaded_document.name, payload)
                st.session_state.analysis_text = extracted.text
                st.session_state._document_status = (
                    uploaded_document.name,
                    extracted.original_characters,
                    extracted.truncated,
                    None,
                )
            except UserInputError as exc:
                st.session_state._document_status = (uploaded_document.name, 0, False, str(exc))

        status = st.session_state.get("_document_status")
        if status and status[0] == uploaded_document.name:
            _, characters, truncated, error = status
            if error:
                st.error(error)
            else:
                st.success(f"Texto extraído: {characters:,} caracteres. Revísalo antes de clasificar.")
                if truncated:
                    st.warning(
                        f"El documento excede {MAX_TEXT_CHARACTERS:,} caracteres; se cargó el inicio para mantener una clasificación estable."
                    )

else:
    st.markdown("**Dictar el texto**")
    st.caption(
        "El navegador solicitará permiso para usar el micrófono. Al detener el dictado, el audio se procesará "
        "mediante el reconocimiento de voz de Google; revisa el texto antes de clasificarlo."
    )
    dictated_text = speech_to_text(
        language="es-CO",
        start_prompt="Iniciar dictado",
        stop_prompt="Detener y transcribir",
        just_once=True,
        use_container_width=True,
        key="ods_dictation",
    )
    if dictated_text:
        normalized_dictation = dictated_text.strip()
        if normalized_dictation and normalized_dictation != st.session_state.get("_last_dictation"):
            st.session_state.analysis_text = normalized_dictation[:MAX_TEXT_CHARACTERS]
            st.session_state._last_dictation = normalized_dictation
            st.success("Dictado transcrito. Revisa el texto antes de clasificarlo.")

left, right = st.columns([1.55, 1], gap="large")
with left:
    text_input = st.text_area(
        "Texto para analizar",
        key="analysis_text",
        height=112,
        max_chars=MAX_TEXT_CHARACTERS,
        placeholder="Describe aquí una política, problema social, iniciativa ambiental o argumento relacionado con el desarrollo sostenible…",
        help=f"Para obtener una señal más estable, utiliza al menos una oración completa. Máximo {MAX_TEXT_CHARACTERS:,} caracteres.",
    )
    analyze = st.button("Clasificar el texto", type="primary", use_container_width=True)

with right:
    result_slot = st.container()
    if not analyze:
        result_slot.markdown(
            '<div class="empty-result">El dictamen aparecerá aquí después de analizar el texto.</div>',
            unsafe_allow_html=True,
        )

if analyze:
    cleaned = text_input.strip()
    if not cleaned:
        st.error("El campo está vacío. Escribe o selecciona un texto antes de clasificar.")
    elif len(cleaned.split()) < 5:
        st.warning("El texto es demasiado breve para ofrecer una señal útil. Añade una oración con más contexto.")
    else:
        try:
            model = load_model()
            prediction = int(model.predict([cleaned])[0])
            ranked = alternatives(model, cleaned)
            with result_slot:
                st.markdown(
                    f'<div class="result"><div class="label">ODS predominante</div>'
                    f'<div class="number">{prediction:02d}</div>'
                    f'<div class="name">{ODS_NAMES[prediction]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.subheader("Alternativas del clasificador")
            st.caption("Los valores son márgenes de decisión de la SVM; indican orden relativo, no probabilidades.")
            alternatives_html = ['<div class="alt-list">']
            for ods, margin, scale in ranked:
                alternatives_html.append(
                    '<div class="alt-item">'
                    f'<div class="alt-heading"><span class="alt-title">ODS {ods} · {ODS_NAMES[ods]}</span>'
                    f'<span class="alt-margin">{margin:+.2f}</span></div>'
                    f'<p class="alt-description">{ODS_DETAILS[ods]}</p>'
                    f'<div class="track"><div class="fill" style="width:{max(5, scale * 100):.1f}%"></div></div>'
                    '</div>'
                )
            alternatives_html.append('</div>')
            st.markdown("".join(alternatives_html), unsafe_allow_html=True)
            st.markdown(
                '<div class="note"><strong>Cómo leer este resultado.</strong> Úsalo como apoyo exploratorio. '
                "El modelo puede confundir objetivos con vocabulario cercano y no reemplaza una revisión temática experta.</div>",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"No fue posible ejecutar el modelo. Verifica los archivos del proyecto. Detalle: {exc}")

with st.expander("Alcance y desempeño del modelo"):
    st.markdown(
        """
        - **Cobertura:** ODS 1 a 16. El conjunto de datos no incluye observaciones del ODS 17.
        - **Evaluación independiente:** exactitud 0,8727; F1 macro 0,8406; F1 ponderado 0,8727.
        - **Pipeline:** limpieza reproducible, TF-IDF con unigramas y bigramas, SVD de 300 componentes, normalización y LinearSVC.
        - **Uso previsto:** demostración académica y exploración de textos en español; no es un sistema de decisión automática.
        """
    )

st.markdown(
    '<p class="footer-note">Proyecto académico · Machine Learning no supervisado · El ODS 17 está fuera del alcance por ausencia de datos de entrenamiento.</p>',
    unsafe_allow_html=True,
)
