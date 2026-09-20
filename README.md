# Clasificación de textos según los ODS

Proyecto académico de procesamiento de lenguaje natural en español. Combina análisis semántico latente (LSA) y clasificación multiclase para relacionar textos con los Objetivos de Desarrollo Sostenible 1 a 16.

## Resultados principales

- 9.656 textos y 16 clases.
- Pipeline final: limpieza → TF-IDF → SVD (300 componentes) → normalización → LinearSVC.
- Conjunto de prueba independiente: exactitud 0,8727; F1 macro 0,8406; F1 ponderado 0,8727.
- Limitación: no existen observaciones del ODS 17 en los datos disponibles.

## Entregables

- `proyecto_ods_entrega.ipynb`: versión compacta recomendada para la entrega, ejecutada y orientada a los puntos C, D y F.
- `proyecto_ods_entrega.html`: exportación navegable de la versión compacta.
- `proyecto_ods.ipynb`: notebook completo conservado como respaldo técnico.
- `proyecto_ods.html`: exportación del respaldo completo.
- `models/pipeline_ods.joblib`: pipeline entrenado y serializado.
- `models/metadata_modelo.json`: métricas, parámetros y huellas de archivos.
- `data/particion_train_test.csv`: partición reproducible de entrenamiento y prueba.
- `app.py`: aplicación Streamlit para clasificar textos escritos, adjuntos o dictados.

## Instalación

Desde el directorio del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación muestra el ODS predicho y las principales alternativas. Los valores del clasificador son márgenes de decisión relativos de `LinearSVC`; no son probabilidades calibradas.

### Adjuntar documentos y dictar

La interfaz admite documentos TXT, MD, PDF y DOCX de hasta 5 MB. Los PDF deben contener texto seleccionable; los documentos escaneados requieren OCR antes de adjuntarlos.

El dictado utiliza el micrófono del navegador con `st.audio_input` y transcribe el audio con OpenAI Speech-to-Text (`gpt-4o-mini-transcribe`). Debes configurar `OPENAI_API_KEY` en Streamlit Cloud (Secrets) o como variable de entorno para habilitar esta función.

## Reproducir el análisis

Abrir `proyecto_ods_entrega.ipynb` y ejecutar todas las celdas desde la raíz del proyecto. Para regenerar el HTML recomendado:

```bash
jupyter nbconvert --to html proyecto_ods_entrega.ipynb
```

Las rutas utilizadas por el notebook son relativas al proyecto y las semillas aleatorias están fijadas.
