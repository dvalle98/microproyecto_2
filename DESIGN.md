# Sistema visual de Brújula ODS

## Dirección

La aplicación adopta la forma de una **mesa de análisis**: el texto funciona como evidencia de entrada y el ODS como dictamen. La composición evita el tablero convencional de métricas y prioriza una única tarea verificable.

## Paleta

- `#10263D` — tinta principal y superficie del dictamen.
- `#F3F1E8` — papel de fondo.
- `#FFFDF5` — hoja de escritura.
- `#13795B` — verde mineral para acción y comparación.
- `#F4C542` — señal amarilla reservada para orientación y resultado.
- `#526474` y `#B7C0B9` — texto secundario y reglas.

La información no depende solamente del color: números, nombres y etiquetas acompañan cada estado.

## Tipografía y jerarquía

La familia de interfaz es sans serif, con `Avenir Next` como voz preferida y alternativas del sistema. El título usa peso alto y espaciado compacto; el cuerpo mantiene una medida cercana a 68 caracteres. Los márgenes de decisión usan numerales tabulares.

## Componentes

- **Origen de entrada:** control compacto para escribir, adjuntar o dictar sin duplicar el flujo principal.
- **Entrada:** hoja clara con borde de tinta, foco verde y área suficiente para revisar o corregir el texto, cualquiera que sea su origen.
- **Acción:** botón de tinta con texto blanco; en interacción cambia a verde mineral y conserva un foco amarillo visible.
- **Dictamen:** placa de tinta con regla amarilla, número ODS dominante y nombre completo.
- **Alternativas:** lista separada por reglas con número, nombre, explicación breve, barra comparativa y margen numérico; nunca se denominan probabilidades.
- **Notas:** superficies verdes suaves para interpretación y límites.

Los radios se mantienen entre 10 y 14 px. La profundidad proviene de sombras suaves con desplazamiento, no de halos.

## Movimiento y estados

El único gesto de entrada es el despliegue breve del dictamen mediante recorte y desenfoque. Se desactiva con `prefers-reduced-motion`. La interfaz incluye estados vacío, validación por texto ausente o demasiado corto, extracción correcta o fallida, transcripción pendiente, carga del modelo y errores recuperables de archivo, credencial, red o cuota.

## Responsividad

En escritorio, entrada y dictamen comparten una composición de dos columnas. En móvil, Streamlit los apila; el título, el método de cuatro pasos y los controles reducen su separación para acercar la acción al primer viewport.

## Límites duraderos

- No presentar los márgenes de `LinearSVC` como probabilidades.
- Mantener visible que el ODS 17 está fuera del alcance.
- No sustituir el flujo principal por una colección de tarjetas o indicadores decorativos.
- Preservar contraste, navegación por teclado y lectura móvil.
- Nunca clasificar automáticamente un documento o dictado: el texto resultante debe quedar visible y editable antes de la acción.
- Explicar en el punto de uso cuándo una grabación se enviará al servicio de transcripción.
