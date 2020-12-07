# A continuación importaremos varias bibliotecas que se utilizarán:

# Biblioteca para trabajar con JSON
import json

import pandas as pd
# Biblioteca con métodos numéricos y representaciones matriciales
import numpy as np
# Biblioteca para construir un modelo basado en la técnica Gradient Boosting
import xgboost as xgb
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import collections
from sklearn.model_selection import train_test_split
# Método para crear modelos basados en árboles de decisión
from sklearn.tree import DecisionTreeClassifier
# Clase para crear una pipeline de machine-learning
from sklearn.pipeline import Pipeline
# Paquetes scikit-learn para evaluación de modelos
# Métodos para la validación cruzada del modelo creado
from sklearn.model_selection import KFold, cross_validate
from IPython.display import display  # display from IPython.display
import itertools
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.pyplot import figure

import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from datetime import date, timedelta


def homogenizar_cat(x):
    caracteres_delete = ["de", " y ", ",", "_", "-", " ", ".", "/", "%"]
    x = x.lower()
    for caract in caracteres_delete:
        x = x.replace(caract, "")
    return x.upper()


def add_dataTime(data):
    data = data.copy()

    def create_data(x):
        tiempo = date(int(x["AÑO"]), int(x["MES"]), 1)
        return pd.Series([tiempo], index=['FECHA'])
    data = data.join(data.apply(create_data, axis=1))
    data = data.drop(columns=["AÑO", "MES"])
    return data


def print_valores_en_fecha(data, x_label="FECHA", y_label="DEMANDA_VENTAS", leyenda="PRODUCTO"):
    data = data.sort_values(by=x_label)
    snplot = sns.catplot(x=x_label, y=y_label, hue=leyenda, kind="point",
                         data=data[[x_label, y_label, leyenda]], height=5,
                         aspect=2.5, title="{} VS {}".format(y_label, x_label), palette=sns.color_palette("colorblind"))

    for ax in snplot.axes.flat[:2]:
        ax.tick_params(axis='x', labelrotation=90)


def Fechas_totales(data):
    data = data.copy()
    min_fecha = min(data.FECHA.unique().tolist())
    max_fecha = max(data.FECHA.unique().tolist())
    fechas_total = []
    curr = min_fecha
    while curr <= max_fecha:
        fechas_total.append(curr)
        mes = curr.month
        anio = curr.year
        if mes < 12:
            mes += 1
        else:
            mes = 1
            anio += 1
        curr = date(int(anio), int(mes), 1)
    return fechas_total


def completa_fechas(data0, lista_productos, fechas):
    data = data0.copy()
    for producto in lista_productos:
        dataprod = data0.loc[data0["PRODUCTO"] == producto]
        for fecha in fechas:
            dataprod0 = dataprod.loc[dataprod["FECHA"] == fecha]
            if len(dataprod0) == 0:
                df = pd.DataFrame(
                    data=[[producto, fecha, 0]], columns=list(data0.columns))
                data = data.append(df, ignore_index=True)
    data = data.sort_values(by="FECHA")
    return data


def graficar_corr_btn_product(data):
    sns_g = sns.pairplot(data, kind="reg", palette="Set1",
                         corner=True)  # TRACK_DH
    sns_g.fig.suptitle("Correlación entre productos")
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 8))
    corr_df = data.corr(method='pearson')
    matrix = np.triu(corr_df)
    hmap = sns.heatmap(corr_df, annot=True, ax=ax,
                       mask=matrix, vmin=-1, vmax=1, center=0)


def add_roll_agg(data, Wmean=None, Wstd=None, Wpct=None, dropna=False):
    """
    Introducir tabla con los productos en el encabezado y con los valores de tiempo ordenados como valores
    Retorna:
    Los datos agregados en nuevas columnas cone el prefijo _mean, _std o _pct
    """
    data = data.copy()
    data = data.reset_index(drop=True)
    # db_work_nofecha.join(db_work_nofecha.shift(1), rsuffix="_t2")
    if Wmean:
        rolmean = data.rolling(window=Wmean,).mean()
    if Wstd:
        rolstd = data.rolling(window=Wstd,).std()
    if Wpct:
        pct = data.pct_change(Wpct)

    if Wmean:
        data = data.join(rolmean, rsuffix="_mean")
    if Wstd:
        data = data.join(rolstd, rsuffix="_std")
    if Wpct:
        data = data.join(pct, rsuffix="_pct")
    if dropna:
        data = data.dropna()
    return data


def gen_time_series(data, steps=3, dropna=False):
    data = data.copy()
    delays = [data.shift(t_delay) for t_delay in range(1, steps+1)]
    for ti, delayTable in enumerate(delays, 1):
        data = data.join(delayTable, rsuffix="_"+str(ti))

    if dropna:
        data = data.dropna()
    return data


def print_forecasting(predicted, trueval, productos):
    for column_name in productos:
        _ = plt.plot(predicted[[column_name]],
                     color='blue', label=column_name+" model",)
        __ = plt.plot(trueval[[column_name]], color='red',
                      label=column_name+" true")
        rsqr = r2_score(trueval[[column_name]], predicted[[column_name]])
        plt.legend(loc='best')
        plt.title('Forecasting Demanda - {} - r2 {}'.format(column_name, rsqr))
        plt.show(block=False)


def print_wholedata_forecasting(inferencia, trueval, productos, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    for column_name in productos:
        _ = plt.plot(inferencia["Fecha"], inferencia[column_name],
                     color='blue', label=column_name+" model",)
        __ = plt.plot(trueval["Fecha"], trueval[column_name],
                      color='red', label=column_name+" true")
    rsqr = r2_score(trueval[[column_name]], inferencia[[column_name]])
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.title('Forecasting Demanda - {} - r2 {}'.format(column_name, rsqr))
    plt.show(block=False)
