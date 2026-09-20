# Producto

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegado por el usuario al solicitar completar el proyecto: Streamlit sobre el pipeline existente de Python y scikit-learn.

## Users

Estudiantes, docentes o evaluadores que necesitan comprobar de forma rápida cómo el modelo clasifica un texto en español dentro de los Objetivos de Desarrollo Sostenible (ODS).

## Product Purpose

Permitir ingresar un texto libre, ejecutar exactamente el pipeline entrenado en el proyecto y comunicar el ODS predicho con suficiente contexto para interpretar el resultado. El éxito consiste en que la predicción sea fácil de probar, comprensible y fiel a las limitaciones del modelo.

## Positioning

La interfaz no es un clasificador genérico: reutiliza el modelo reproducible y evaluado en el notebook del proyecto, incluida la misma limpieza, representación TF-IDF, reducción LSA y clasificación SVM.

## Operating Context

Se ejecuta como demostración académica en Streamlit. El usuario redacta un texto, extrae el contenido de un documento o transcribe un dictado en español, revisa el texto resultante, solicita la clasificación y consulta el resultado principal y las alternativas según el margen del clasificador.

## Capabilities and Constraints

- Clasifica únicamente entre los ODS 1 a 16.
- El conjunto disponible no contiene ejemplos del ODS 17.
- `LinearSVC` produce márgenes de decisión, no probabilidades calibradas.
- El modelo se carga desde `models/pipeline_ods.joblib` y requiere el módulo local `src.text_processing`.
- Los documentos admitidos son TXT, MD, PDF y DOCX de hasta 5 MB; los PDF escaneados requieren OCR previo.
- El dictado utiliza el micrófono del navegador y el reconocimiento de voz de Google mediante `streamlit-mic-recorder`; no requiere una clave propia y depende de conexión a internet y compatibilidad del navegador.
- Las métricas del conjunto de prueba son: exactitud 0,8727, F1 macro 0,8406 y F1 ponderado 0,8727.

## Evidence on Hand

- Notebook ejecutado: `proyecto_ods.ipynb`.
- Pipeline serializado: `models/pipeline_ods.joblib`.
- Metadatos y métricas: `models/metadata_modelo.json`.
- Datos fuente: `data/_94f47dfe96414485b5def55bccbd7cb8_Datos_textosODS.xlsx`.
- No hay testimonios, acreditaciones ni afirmaciones de uso en producción; no deben inventarse.

## Product Principles

1. Mostrar la predicción sin exagerar la certeza del modelo.
2. Mantener trazabilidad entre interfaz, notebook y pipeline serializado.
3. Explicar las limitaciones relevantes en el punto de uso.
4. Favorecer una interacción breve y verificable.

## Accessibility & Inclusion

La interfaz debe funcionar con teclado, mantener contraste legible, no depender únicamente del color y adaptarse a pantallas móviles.
