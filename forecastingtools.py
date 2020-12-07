# A continuación importaremos varias bibliotecas que se utilizarán:
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
from sklearn.preprocessing import MinMaxScaler
from utiles import *
from customlayers import model_forecasting


def gen_input(data, window_agg, n_steps, productos):
    data = data.copy()
    scaler_y = MinMaxScaler()
    scaler_x = MinMaxScaler()

    # ajustando el scaler para los datos que se serán de salida
    scaler_y.fit(data)

    # Generando los datos agregados: Media movil y la STD movil
    data = add_roll_agg(data, Wmean=window_agg, Wstd=window_agg, dropna=True)
    Y_columns = list(data.columns)

    # Generando los delays de tiempo
    # 11 delays + la actual posicion = 12
    data = gen_time_series(data, steps=n_steps-1, dropna=True)
    data = data.reset_index(drop=True)

    # Escalando los datos
    X = data
    scaler_x.fit(X)
    X = scaler_x.transform(X)

    x_input = X[-1]
    x_input = np.expand_dims(x_input, axis=0)

    # Reshape vector
    n_features = int(len(productos)*3)  # len(productos) * 3(valor, mean, std)
    x_input = x_input.reshape((-1, n_steps, n_features))
    input_shape = x_input.shape[1:]

    return x_input, input_shape, scaler_x, scaler_y


def forecasting_n(productos, product=None, n_ahead=None, n_steps=12, window_agg=3, data_base=None, model_path=None):
    # Escaladores de datos

    # Abriendo el archivo
    df_study = pd.read_excel("data/DB_Data_Demand.xlsx", sheet_name="DATA")
    col_interes = ["Año", "Mes", "Product_Code", "Demand"]
    new_col_names = ["AÑO", "MES", "PRODUCTO", "DEMANDA_VENTAS"]
    df_study = df_study[col_interes]
    df_study.columns = new_col_names
    df_study["PRODUCTO"] = df_study["PRODUCTO"].apply(homogenizar_cat)
    df_study_mod = add_dataTime(df_study.copy())
    df_study_mod = df_study_mod.loc[df_study_mod["PRODUCTO"].isin(
        productos)]  # filtrando solo los productos de myor presencia

    df_study_mod = df_study_mod.groupby(
        by=["PRODUCTO", "FECHA"], as_index=False).mean().sort_values(by="FECHA")
    df_study_mod = df_study_mod.reset_index(drop=True)
    # df_study_mod = df_study_mod.iloc[:15]         # Selecciona  los 15 ultimos dias
    df_study_mod["PRODUCTO"] = df_study_mod["PRODUCTO"].astype(str)

    fechas_total = Fechas_totales(df_study_mod)
    df_study_mod = completa_fechas(df_study_mod, productos, fechas_total)
    # Pivotear las columnas para obtener las series de tiempo
    df_study_mod = df_study_mod.pivot(
        index="FECHA", columns=["PRODUCTO"], values=["DEMANDA_VENTAS"])
    df_study_mod = df_study_mod.sort_values(by="FECHA")
    df_study_mod.columns = df_study_mod.columns.droplevel()
    Lista_productos_ord = list(df_study_mod.columns)
    # Input shape
    input_shape = (n_steps, int(len(productos)*3))

    # Cargando modelo guardado
    model = model_forecasting(input_shape, n_outputs=len(
        Lista_productos_ord), saved_file=model_path, summary=False)

    if n_ahead == 1:
        # Manipulamos la tabla total, transformamos y extraemos el ultimo dato
        x_input, input_shape, scaler_x, scaler_y = gen_input(
            df_study_mod, window_agg, n_steps, Lista_productos_ord)
        forescast = model.predict(x_input)
        forescast_scal = scaler_y.inverse_transform(forescast)
        forescast_df = pd.DataFrame(
            forescast_scal, columns=Lista_productos_ord)
        return forescast_df

    # Forecasting n ahead
    for _ in range(n_ahead):
        # Manipulamos la tabla total, transformamos y extraemos el ultimo dato
        x_input, input_shape, scaler_x, scaler_y = gen_input(
            df_study_mod, window_agg, n_steps, Lista_productos_ord)

        # Forecasting
        forescast = model.predict(x_input)
        forescast_scal = scaler_y.inverse_transform(forescast)

        # Dataframe Forecating
        forescast_df = pd.DataFrame(
            forescast_scal, columns=Lista_productos_ord)
        df_study_mod = df_study_mod.append(forescast_df, ignore_index=True)

    # Entregando producto en particular
    if product:
        return df_study_mod.iloc[-n_ahead:, Lista_productos_ord.index(product)]
    return df_study_mod.iloc[-n_ahead:]
