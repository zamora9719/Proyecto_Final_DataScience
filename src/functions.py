import pandas as pd
from scipy.stats import anderson
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from sklearn.model_selection import (
    cross_validate,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline

#==============================================================================
# FUNCIONES DE DATA MINING
#==============================================================================

def unique_col_values(df):
    """
    Función que printea 3 dimensiones de un DataFrame:
    1. Nombre de las columnas que contiene
    2. Cantidad de valores únicos que hay en cada columna
    3. Tipo de datos que tiene cada columna

    Args:
       df: DataFrame que contiene las variables
    """

    for column in df:
        print("{} | {} | {}".format(
            df[column].name, len(df[column].unique()), df[column].dtype
        ))

def test_anderson_norm(df):
   """
   Función que calcula el test de normalidad Anderson-Darling para las
   variables de un DataFrame.

   Args:
       df: DataFrame que contiene las variables

   Returns:
       Diccionario con los resultados
   """
   resultados = {}
   cols = df.select_dtypes(include=[np.number]).columns
   for col in cols:
       resultado = anderson(df[col].dropna(), dist='norm')
       resultados[col] = {
           'statistic': resultado.statistic,
           'critical_values': resultado.critical_values,
           'significance_level': resultado.significance_level
       }

   return resultados

def heatmap_corr(df, metodologia):
   """
   Función que representa un mapa de calor para el análisis de correlaciones.
   Genera también una máscara previa para devolver solamente la parte que queda
   por debajo de la diagonal principal.

   Args:
       df: DataFrame que contiene las variables
       metodologia: permite elegir entre calcular la matriz de correlación por
       el método de Pearson (requiere normalidad) o de rangos de Spearman.
   """
   corr = df.corr(method=metodologia)
   mascara = np.triu(np.ones_like(corr, dtype=bool))
   sns.heatmap(corr, cmap='coolwarm', annot=True,
               fmt='.2f', annot_kws={'size': 8}, mask=mascara)
   plt.xlabel(f'Mapa de calor de la matriz de correlación de {metodologia}')
   plt.show()

def reset_test(df, X, y, potencia):
    """
    Función que efectúa el test RESET de Ramsey para evaluar la presencia de
    relaciones polinómicas en los datos.

    Args:
        X: matriz de variables independientes
        y: variable objetivo
        potencia: exponente para el cual se desea realizar la comprobación

    Returns:
        Estadístico y p-valor de rechazar la hipótesis nula (H0: no existen
        relaciones polinómicas en los datos) para la potencia elegida
    """
    df_limpio = df.dropna()
    if len(df_limpio) >= 100:
        pass
    else:
        raise ValueError(f'Después de eliminar nulos, los registros son \
        demasiado pocos como para que el test tenga validez.')
    X_test = df_limpio[X]
    X_const = sm.add_constant(X_test)
    y_test = df_limpio[y]
    model = sm.OLS(y_test, X_const).fit()
    reset_test = linear_reset(model, power=potencia, test_type='fitted')
    return reset_test
   
def grafico_caja(df, column):
   """
   Función que representa un boxplot para el análisis de datos atipicos.

   Args:
       df: DataFrame que contiene las variables
       column: variable que queremos representar
   """
   sns.boxplot(data=df, y=column)
   plt.xlabel(f'Boxplot de {column}')
   plt.show()

#==============================================================================
# FUNCIONES DE MODEL SELECTION y HIPERPARAMS OPTIMIZATION
#==============================================================================

def cv_modelos(modelos, prep, selector_lasso, X, y):
    """
    Función que realiza validación cruzada sobre el conjunto de modelos que se le
    pasen como argumento.

    Args:
        modelos: Diccionario compuesto de una clave, que se corresponde con el 
        alias de cada modelo; y un valor, que consiste en el modelo instanciado.
        prep: pipeline de preprocesado de los datos que se quiere utilizar en
        la cv.
        selector_lasso: modelo selector de variables L1 que se haya utilizado.
        X: datos asociados a las variables independientes del dataset.
        y: datos asociados a la variable objetivo del dataset.

    Returns:
        Diccionario con los resultados de las distintas métricas que utiliza la
        función para evaluar los modelos (r2, mae, rmse, r2_ratio, fit_time)
    """
    resultados={}
    
    for modelo_nombre, modelo in modelos.items():
        
        pipeline_completo = Pipeline([
        ('prep', prep),
        ('selector', selector_lasso),
        ('modelo', modelo)
        ])
        
        resultados_cv = cross_validate(
            estimator=pipeline_completo,
            X=X,
            y=y,
            cv=5,
            n_jobs=-1,
            return_train_score=True,
            error_score='raise',
            scoring={
            'r2': 'r2',
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
            }
            )

        resultados[modelo_nombre] = {
            "r2_test": float(resultados_cv["test_r2"].mean()),
            "rmse_test": float(-resultados_cv["test_rmse"].mean()),
            "mae_test": float(-resultados_cv["test_mae"].mean()),
            "r2_test_var": float(resultados_cv["test_r2"].var()),
            "rmse_test_var": float(resultados_cv["test_rmse"].var()),
            "mae_test_var": float(resultados_cv["test_mae"].var()),
            "r2_train": float(resultados_cv["train_r2"].mean()),
            "rmse_train": float(-resultados_cv["train_rmse"].mean()),
            "mae_train": float(-resultados_cv["train_mae"].mean()),
            "fit_time": float(resultados_cv["fit_time"].mean()),
            "r2_ratio": float(resultados_cv["train_r2"].mean() /
                                  max(resultados_cv["test_r2"].mean(), 0.001))
        }

    return resultados

def optimizar_hiperparametros(dict_modelos_parametros, prep, selector_lasso, X, y, seed):
    """
    Función que optimiza los hiperparámetros de los modelos que 
    se pasan como argumento. 

    Args:
        dict_modelos_parametros: Diccionario cuya clave es el
        alias de cada modelo a optimizar, y el valor es una 
        tupla compuesta del modelo instanciado + otro diccionario
        con los hiperparámetros a optimizar.
        prep: pipeline de preprocesado de los datos que se quiere utilizar en
        la cv.
        selector_lasso: modelo selector de variables L1 que se haya utilizado.
        X: datos asociados a las variables independientes del dataset.
        y: datos asociados a la variable objetivo del dataset.
        seed: semilla aleatoria para reproducibilidad.


    Returns:
        Diccionario con los hiperparámetros optimizados para cada
        modelo, y resultado de las métricas empleadas por la función
        para evaluar los modelos (r2, rmse, mae, r2_gap, fit_time).
    """
    optimizacion = {}
    
    for model_name, (model, params) in dict_modelos_parametros.items():
        
        pipeline_completo = Pipeline([
            ('prep', prep),
            ('selector', selector_lasso),
            ('modelo', model)
            ])

        resultados_opt = RandomizedSearchCV(estimator=pipeline_completo,
                                            param_distributions=params,
                                            scoring={
                                                'r2': 'r2',
                                                'rmse': 'neg_root_mean_squared_error',
                                                'mae': 'neg_mean_absolute_error'
                                                },
                                            cv=5,
                                            n_iter=50,
                                            n_jobs=-1,
                                            random_state=seed,
                                            refit='r2',
                                            return_train_score=True)

        resultados_opt.fit(X=X, y=y)

        best_idx = resultados_opt.best_index_

        optimizacion[model_name] = {
            'mejor_estimador': resultados_opt.best_estimator_,
            'mejores_parametros': resultados_opt.best_params_,
            'mejor_r2': resultados_opt.best_score_,
            'mejor_rmse_test': -resultados_opt.cv_results_['mean_test_rmse'][best_idx],
            'mejor_mae_test': -resultados_opt.cv_results_['mean_test_mae'][best_idx],
            'mejor_r2_train': resultados_opt.cv_results_['mean_train_r2'][best_idx],
            'mejor_rmse_train': -resultados_opt.cv_results_['mean_train_rmse'][best_idx],
            'mejor_mae_train': -resultados_opt.cv_results_['mean_train_mae'][best_idx],
            'r2_var_test': (resultados_opt.cv_results_['std_test_r2'][best_idx]) ** 2,
            'r2_var_train': (resultados_opt.cv_results_['std_train_r2'][best_idx]) ** 2,
            'rmse_var_test': (resultados_opt.cv_results_['std_test_rmse'][best_idx]) ** 2,
            'rmse_var_train': (resultados_opt.cv_results_['std_train_rmse'][best_idx]) ** 2,
            'r2_gap': resultados_opt.cv_results_['mean_train_r2'][best_idx] - resultados_opt.best_score_
        }

    return optimizacion