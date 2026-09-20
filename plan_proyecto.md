# Plan de trabajo: clasificación de textos según los ODS

## Estado de avance

- [x] Fase 1. Carga, auditoría y comprensión de los datos.
- [x] Fase 2. Separación estratificada de entrenamiento y prueba con control de grupos similares.
- [x] Fase 3. Preparación del texto y representación TF-IDF.
- [x] Fase 4. Modelo LSA e interpretación de tópicos.
- [x] Fase 5. Modelos de clasificación.
- [x] Fase 6. Búsqueda de hiperparámetros y validación.
- [x] Fase 7. Evaluación final.
- [x] Fase 8. Documentación y entregables.
- [x] Fase 9. Aplicación Streamlit opcional.

## 1. Lectura del problema

El proyecto debe resolver un problema de **clasificación multiclase de texto**: dado un texto en español, asignarle un Objetivo de Desarrollo Sostenible (ODS). Además del modelo predictivo, se debe construir un modelo de tópicos mediante análisis semántico latente (LSA) y relacionar cualitativamente al menos cinco componentes con los ODS.

El entregable principal será un *notebook* reproducible en formatos `.ipynb` y `.html`, con las salidas de todas las celdas visibles, las decisiones justificadas y al menos cuatro predicciones sobre textos que no hayan participado en el entrenamiento.

## 2. Diagnóstico inicial de los datos

El archivo disponible es `data/_94f47dfe96414485b5def55bccbd7cb8_Datos_textosODS.xlsx` y contiene:

- 9.656 observaciones.
- Dos columnas: `textos` y `ODS`.
- Ningún valor nulo.
- Ningún texto duplicado de forma exacta.
- Longitud media aproximada de 709 caracteres por texto.
- 16 clases, correspondientes a los ODS 1 a 16.
- Distribución desigual entre clases: la clase más frecuente tiene 1.080 textos y la menos frecuente, 312.

### Limitación principal

El conjunto no contiene ejemplos del **ODS 17**. Por tanto, con estos datos no es posible entrenar ni evaluar un modelo capaz de predecir dicho ODS. El alcance empírico del proyecto será la clasificación entre los ODS 1 a 16, salvo que se consiga una fuente adicional, compatible y etiquetada para el ODS 17.

## 3. Estrategia general

La solución se organizará en dos líneas relacionadas:

1. **Análisis no supervisado:** TF-IDF + `TruncatedSVD` para construir el modelo LSA e interpretar tópicos.
2. **Clasificación supervisada:** TF-IDF + reducción de dimensionalidad + clasificador multiclase, con búsqueda de hiperparámetros y evaluación sobre un conjunto de prueba independiente.

Todas las transformaciones aprendidas —vocabulario TF-IDF, pesos IDF, componentes SVD y clasificador— se encapsularán en un `Pipeline` de scikit-learn. Esto evita fuga de información durante la validación cruzada y permite procesar textos nuevos de forma consistente.

## 4. Fases de desarrollo

### Fase 1. Carga, auditoría y comprensión de los datos

Objetivos:

- Cargar el archivo y revisar tipos, dimensiones, valores faltantes y duplicados.
- Confirmar el conjunto real de etiquetas.
- Calcular la distribución absoluta y porcentual de los ODS.
- Analizar la longitud de los textos y detectar casos atípicos.
- Revisar una pequeña muestra por clase para identificar problemas de traducción, textos sintéticos o ruido.
- Buscar duplicados aproximados o textos casi idénticos, ya que la aumentación con ChatGPT podría generar observaciones muy similares.

Salidas previstas:

- Tabla resumen de calidad de datos.
- Gráfico de distribución de clases.
- Gráfico o tabla de longitud de los textos.
- Declaración explícita sobre la ausencia del ODS 17.

### Fase 2. Separación de entrenamiento y prueba

Antes de ajustar TF-IDF o cualquier modelo, se reservará un conjunto de prueba mediante una partición estratificada:

- 80 % para desarrollo y entrenamiento.
- 20 % para prueba final.
- `random_state` fijo para reproducibilidad.
- Estratificación por `ODS` para conservar la proporción de clases.

El conjunto de prueba no se utilizará en análisis que aprendan parámetros, selección de modelos ni búsqueda de hiperparámetros. Si se detectan grupos de textos casi duplicados, la partición se realizará por grupos para evitar que versiones muy similares del mismo texto queden a ambos lados de la división.

### Fase 3. Preparación del texto y representación TF-IDF

Se comenzará con un preprocesamiento moderado, porque una limpieza agresiva puede eliminar información útil:

- Conversión a minúsculas mediante `TfidfVectorizer`.
- Normalización de espacios.
- Conservación inicial de tildes y caracteres españoles.
- Eliminación de signos mediante el patrón de tokenización del vectorizador.
- Evaluación de *stopwords* en español como hiperparámetro, en lugar de asumir que siempre ayudan.
- Evaluación de unigramas frente a unigramas + bigramas.
- Control de términos muy raros y muy frecuentes con `min_df` y `max_df`.
- Uso opcional de `sublinear_tf=True`.

No se aplicará lematización de entrada. Solo se incorporará si una prueba controlada demuestra una mejora suficiente para justificar su complejidad y coste.

Parámetros candidatos:

- `ngram_range`: `(1, 1)` y `(1, 2)`.
- `min_df`: 2, 3 y 5.
- `max_df`: 0,90, 0,95 y 1,0.
- `max_features`: sin límite o un límite razonable según memoria y tiempo.
- `sublinear_tf`: `True` y `False`.
- `stop_words`: sin eliminación o lista española documentada.

### Fase 4. Modelo LSA e interpretación de tópicos

Se ajustará `TruncatedSVD` sobre la matriz TF-IDF del conjunto de entrenamiento.

Procedimiento:

1. Explorar entre 10 y 20 componentes, como exige el enunciado.
2. Reportar la varianza explicada individual y acumulada.
3. Extraer, para cada componente seleccionado, los términos con mayor peso positivo y, cuando aporte claridad, los de mayor peso negativo.
4. Presentar al menos cinco componentes como tópicos.
5. Asignar una interpretación humana a cada tópico y compararlo con uno o varios ODS.
6. Aclarar que los componentes de LSA no corresponden necesariamente de forma uno a uno con las etiquetas ODS.

La interpretación se apoyará en:

- Palabras con mayores *loadings*.
- Textos con mayor puntuación en cada componente.
- Comparación entre las etiquetas reales predominantes dentro de esos textos.

Esto permitirá una interpretación más sólida que usar únicamente listas de palabras.

### Fase 5. Modelos de clasificación

Se trabajará con un enfoque incremental.

#### 5.1 Modelo base

Un modelo simple establecerá una referencia mínima:

- `DummyClassifier(strategy="most_frequent")` o `strategy="stratified"`.

#### 5.2 Modelo principal con reducción de dimensionalidad

Pipeline candidato:

```text
texto → TF-IDF → TruncatedSVD → Normalizer → clasificador
```

Clasificadores candidatos:

- **Regresión logística multinomial:** ofrece una base sólida, probabilidades y parámetros interpretables.
- **SVM lineal:** suele funcionar bien en clasificación de texto; después de SVD puede utilizarse `LinearSVC`.

La primera opción para el modelo final será regresión logística después de LSA, porque facilita mostrar probabilidades o niveles de confianza en una futura aplicación. Se conservará SVM lineal como alternativa si mejora de forma material el F1 macro.

#### 5.3 Comparación de referencia

Como control experimental se puede evaluar un clasificador lineal directamente sobre TF-IDF, sin SVD. Esta comparación mostrará el coste o beneficio predictivo de la reducción de dimensionalidad. Sin embargo, el resultado principal presentado para cumplir la rúbrica incluirá explícitamente una técnica de reducción.

### Fase 6. Búsqueda de hiperparámetros y validación

La búsqueda se realizará únicamente sobre el conjunto de entrenamiento mediante validación cruzada estratificada, preferiblemente con cinco particiones.

Se utilizará `RandomizedSearchCV` o una cuadrícula acotada para explorar conjuntamente:

- Parámetros de TF-IDF.
- Número de componentes SVD.
- Fuerza de regularización del clasificador (`C`).
- Configuración de pesos de clase (`class_weight=None` o `"balanced"`).

Como criterio de selección se usará **F1 macro**, porque da el mismo peso a cada ODS y evita que las clases más grandes dominen la decisión. También se registrará F1 ponderado y exactitud para ofrecer una lectura complementaria.

La búsqueda no debe crecer de forma innecesaria: primero se hará una exploración amplia y económica; después, una búsqueda más estrecha alrededor de las mejores configuraciones.

### Fase 7. Evaluación final

Una vez seleccionados los hiperparámetros, el pipeline completo se reajustará con todos los datos de entrenamiento y se evaluará una sola vez sobre el conjunto de prueba.

Métricas y salidas:

- Exactitud (*accuracy*).
- F1 macro, principal métrica del proyecto.
- F1 ponderado.
- Precisión, *recall* y F1 por clase.
- Matriz de confusión, preferiblemente normalizada por clase real.
- Comparación con el `DummyClassifier`.
- Análisis de errores: pares de ODS que el modelo confunde con mayor frecuencia.
- Al menos cuatro predicciones individuales del conjunto de prueba, mostrando texto, etiqueta real y predicción.

Si el modelo ofrece probabilidades, se puede mostrar la probabilidad de la clase predicha y las tres clases más probables. Esta cifra se describirá como confianza del modelo, no como certeza estadística absoluta.

### Fase 8. Documentación y entregables

El *notebook* seguirá una narrativa reproducible:

1. Contexto y objetivo.
2. Librerías y configuración reproducible.
3. Carga y auditoría de datos.
4. Análisis exploratorio.
5. Separación de entrenamiento y prueba.
6. Preparación de texto y TF-IDF.
7. LSA e interpretación de al menos cinco tópicos.
8. Modelos base y candidatos.
9. Búsqueda de hiperparámetros.
10. Evaluación final.
11. Cuatro o más ejemplos de clasificación.
12. Conclusiones, limitaciones y trabajo futuro.

Antes de entregar:

- Ejecutar el *notebook* completo desde cero.
- Confirmar que todas las celdas muestran su salida.
- Verificar que no existen rutas absolutas dependientes del equipo.
- Fijar semillas aleatorias.
- Exportar el *notebook* a HTML.
- Revisar que tablas y gráficos se visualicen correctamente en ambos formatos.

### Fase 9. Bonificación opcional con Streamlit

La aplicación se desarrollará únicamente después de estabilizar y guardar el pipeline final.

Flujo mínimo:

1. El usuario escribe un texto.
2. La aplicación carga el pipeline serializado.
3. El pipeline aplica exactamente las transformaciones usadas durante el entrenamiento.
4. La aplicación muestra el ODS predicho y, si el modelo lo permite, las principales alternativas.

Archivos previstos para esta fase:

- `app.py`.
- Pipeline entrenado en un archivo serializado.
- `requirements.txt` o equivalente.
- Instrucciones breves de ejecución.

## 5. Correspondencia con la rúbrica

| Requisito | Evidencia prevista |
|---|---|
| Preparación y reducción de dimensionalidad — 30 % | Auditoría, preprocesamiento justificado, TF-IDF, SVD y varianza explicada. |
| Pipeline de preparación — 15 % | `Pipeline` reproducible que encapsula TF-IDF, SVD, normalización y clasificación. |
| Clasificación, hiperparámetros y métricas — 30 % | Modelos candidatos, validación cruzada, búsqueda de hiperparámetros, justificación de F1 macro y evaluación final. |
| Predicciones sobre textos no vistos — 10 % | Tabla con al menos cuatro casos del conjunto de prueba. |
| LSA y cinco tópicos — 15 % | Palabras con mayor peso, textos representativos e interpretación frente a los ODS. |
| Bonificación — 15 puntos | Aplicación Streamlit que reutiliza el pipeline final. |

## 6. Riesgos y controles

| Riesgo | Control propuesto |
|---|---|
| No hay datos del ODS 17 | Declarar el alcance de 16 clases o conseguir datos adicionales antes de entrenar. |
| Desbalance entre ODS | Partición estratificada, F1 macro y prueba de `class_weight="balanced"`. |
| Fuga de información | Reservar prueba al inicio y ejecutar todas las transformaciones dentro del pipeline y la validación cruzada. |
| Textos aumentados muy similares | Buscar similitud aproximada y, si es necesario, dividir por grupos. |
| SVD reduce el rendimiento | Comparar contra TF-IDF sin SVD y justificar el compromiso entre dimensionalidad, interpretación y desempeño. |
| Interpretación subjetiva de tópicos | Combinar *loadings*, textos representativos y distribución de etiquetas reales. |
| Búsqueda costosa | Usar búsqueda aleatoria o una cuadrícula progresiva y acotada. |
| Aplicación distinta al entrenamiento | Serializar y reutilizar el pipeline completo, no solo el clasificador. |

## 7. Orden recomendado de ejecución

1. Completar auditoría y revisión de similitud entre textos.
2. Congelar la partición entrenamiento/prueba.
3. Construir el baseline y el pipeline TF-IDF.
4. Desarrollar e interpretar LSA.
5. Entrenar los modelos candidatos con SVD.
6. Ejecutar la búsqueda de hiperparámetros.
7. Evaluar una sola vez sobre prueba y analizar errores.
8. Redactar conclusiones y producir las cuatro predicciones requeridas.
9. Ejecutar y exportar el *notebook*.
10. Si hay tiempo, construir la aplicación Streamlit.

## 8. Criterio de finalización

El proyecto estará completo cuando el *notebook* pueda ejecutarse de principio a fin, produzca un pipeline entrenado y evaluado sin fuga de información, interprete al menos cinco tópicos LSA, muestre al menos cuatro clasificaciones de prueba, justifique cada decisión relevante y esté disponible en formatos `.ipynb` y `.html`.
