# Clasificación de Riesgo Crediticio en Clientes Bancarios

## Autor
Kevin Gómez Villanueva

## Fecha
10 de diciembre de 2024

---

## Resumen Ejecutivo

Este proyecto implementa un pipeline de Machine Learning para la clasificación de riesgo crediticio en clientes bancarios, utilizando servicios de AWS como Sagemaker y Bedrock. El pipeline abarca:

- **Etiquetado automático de datos** mediante Modelos Grandes de Lenguaje (LLM).
- **Entrenamiento de modelos** de clasificación usando Support Vector Machine.
- **Despliegue del modelo** en un endpoint de Sagemaker para realizar inferencias en tiempo real.

## Índice

1. [Introducción](#introducción)
2. [Objetivos](#objetivos)
3. [Metodología](#metodología)
4. [Desarrollo](#desarrollo)
5. [Conclusiones](#conclusiones)

---

## Introducción

La clasificación de riesgo crediticio es un análisis crucial que ayuda a las instituciones financieras a evaluar la probabilidad de incumplimiento por parte de los clientes. Este proyecto busca optimizar dicho proceso mediante el uso de herramientas avanzadas de Machine Learning y servicios en la nube.

## Objetivos

### Objetivo General
Determinar el nivel de riesgo crediticio de un cliente bancario a partir de sus datos.

### Objetivos Específicos

- Utilizar AWS Bedrock para elaborar descripciones detalladas del perfil del cliente y etiquetar los datos automáticamente.
- Entrenar y desplegar un modelo eficiente en AWS Sagemaker.

## Metodología

### Base de Datos

Se utilizó el archivo `credit_risk_reto.csv`, que contiene 1000 entradas con las siguientes columnas:

- **Age**: Edad del cliente
- **Sex**: Sexo del cliente
- **Job**: Nivel de habilidad laboral (0-3)
- **Housing**: Tipo de alojamiento
- **Saving accounts**: Tipo de cuenta de ahorro
- **Checking account**: Tipo de cuenta corriente
- **Credit amount**: Monto de crédito solicitado
- **Duration**: Duración del préstamo (en meses)
- **Purpose**: Motivo del préstamo

### Etapas del Pipeline

1. **Análisis Exploratorio de Datos (EDA):**
   - Identificación de datos faltantes y desbalanceo de clases.

2. **Etiquetado Automático de Datos:**
   - Uso de AWS Bedrock con modelos como `amazon.titan-text-premier-v1` para generar descripciones detalladas de riesgo.

3. **Preprocesamiento:**
   - Normalización de datos y preparación para entrenamiento.

3. **Entrenamiento de Modelos:**
   - Entrenamiento con diferentes algoritmos de clasificación y selección del mejor modelo basado en métricas de desempeño.

4. **Despliegue:**
   - Implementación del modelo seleccionado en un endpoint de Sagemaker.

5. **Inferencia:**
   - Uso del endpoint para clasificar nuevos datos en tiempo real.

## Desarrollo

Los notebooks y archivos relacionados están disponibles en el siguiente repositorio de GitHub:
[Repositorio del Proyecto](https://github.com/kevinGmezIoT/credit-risk-reto)

### Notebooks Incluidos:

1. **EDA y Preprocesamiento:** Exploración de datos inicial y preparación del dataset.
2. **Etiquetado Automático:** Código para la generación de descripciones usando LLM.
3. **Entrenamiento:** Implementación de modelos de Machine Learning y optimización de hiperparámetros.
4. **Despliegue:** Configuración del endpoint en AWS Sagemaker.
5. **Inferencia:** Pruebas y validación del modelo en producción.

### Archivos de Datos:

- `data/raw/credit_risk_reto.csv`: Dataset inicial.
- `data/processed/credit_risk_reto_preprocessed.csv`: Dataset con valores nulos rellenados.
- `data/processed/output_description.csv`: Descripciones generadas por el LLM.
- `data/processed/output_target.csv`: Clasificación generada por el LLM.
- `data/toTrain/test-V-1.csv`: Dataset de prueba.
- `data/toTrain/train-V-1.csv`: Dataset de entrenamiento.


## Conclusiones

Este proyecto demuestra cómo los servicios en la nube de AWS pueden integrarse para resolver problemas complejos en el sector financiero. El pipeline desarrollado puede adaptarse y escalarse según las necesidades de instituciones bancarias.

## Instalación y Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/kevinGmezIoT/credit-risk-reto.git
   cd credit-risk-reto
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar los notebooks:**
   Abra los notebooks en un entorno como JupyterLab o VSCode y ejecute las celdas en orden.

## Contacto

Si tienes dudas o sugerencias, puedes contactarme en: [kevin.gomez.villanueva.uni@outlook.com](mailto:kevin.gomez.villanueva.uni@outlook.com)
