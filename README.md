# Proyecto Final Programa Experto en Data Science (@[Datamecum]([url](https://datamecum.com/)))

## Naturaleza del problema
En este proyecto, nos enfrentamos a un problema de regresión lineal con un dataset de variables ofuscadas, como los que se suelen ver en plataformas tipo Kaggle o Numerai.

El objetivo consistía en elaborar un modelo de Machine Learning que fuese el mejor predictor posible de la variable objetivo, utilizando para medir la bondad del modelo las métricas típicas en estos casos (R cuadrado, MAE, RMSE, etc.).

En este repositorio encontraréis todo el código que utilicé durante la resolución del problema, así como los ficheros de datos que se nos proporcionaron como inputs, y los que fui generando yo en base a transformaciones de los datos originales.

## Notebooks
La estructura se compone de 3 notebooks:
1. **proyecto_final_EDA**: Análisis exploratorio inicial de los datos, basado en estadísticos relevantes y distribuciones
2. **proyecto_final_seleccion_optimizacion_modelo**: Preprocesamiento de los datos, selección del mejor modelo en base a la información disponible, y ajuste de hiperparámetros de los modelos candidatos
3. **proyecto_final_prediccion**: Implementación de los insights obtenidos en pasos anteriores, en un solo notebook simplificado

## Tecnologías utilizadas
- Python 3.x
- Pandas, NumPy, Scipy, Scikit-learn, Statsmodels, Missingno, IPyWidgets
- LightGBM, CatBoost, XGBoost
- Matplotlib, Seaborn
- VisualStudioCode

## Cómo ejecutar
```bash
pip install -r requirements.txt
jupyter notebook
