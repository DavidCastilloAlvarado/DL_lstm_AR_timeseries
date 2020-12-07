# %%
from forecastingtools import *
from customlayer_v2 import *
import os


class forecasting_demanda(object):
    def __init__(self, producto, n_steps, window_agg, db_path, datos_d_interes, sheetdb, saved_file_model=None, name_model="lstmar", split_train_ratio=0.2, stratif=2):
        self.productos = [producto]
        self.sheetdb = sheetdb
        self.stratif = stratif
        self.datos_d_interes = datos_d_interes
        self.n_steps = n_steps
        self.window_agg = window_agg
        self.db_path = db_path
        self.saved_file = saved_file_model
        self.name_model = name_model
        self.split_train_ratio = split_train_ratio
        self.input_shape = (n_steps, int(len(self.productos)*3))
        self.n_epochs = 0
        self.db_clean = self.readDatafromDB()
        self.X_data, self.Y_data, self.scaler_x, self.scaler_y, self.periodos = self.gen_input()
        print("===================== {} =======================".format(name_model))
        self.model = model_forecasting_v2(input_shape=self.input_shape, n_outputs=len(
            self.productos), saved_file=self.saved_file, summary=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data, self.Y_data, test_size=self.split_train_ratio, random_state=50, stratify=self.periodos["Periodos"])

    def train_model(self, epochs=2500):
        dbsource = self.name_model
        logdir = "logs/"+dbsource
        self.logdir = logdir
        model_filename = dbsource + '.hdf5'
        self.checkpoint_path = os.path.join('model', model_filename)
        epoch_add = epochs
        tboard_callback = TensorBoard(log_dir=logdir)
        # model_checkpoint = ModelCheckpoint('model/'+dbsource+'/LSTMAR_'+dbsource+'.hdf5', monitor='val_loss',verbose=1, save_best_only=True,)
        model_checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path, monitor='val_r2_coeff_det', verbose=1, save_best_only=True, mode="max",)
        earlyStopping = EarlyStopping(
            monitor='val_loss', patience=300, min_delta=0)

        history = self.model.fit(self.X_train, self.y_train,
                                 validation_data=(self.X_test, self.y_test),
                                 epochs=self.n_epochs + epoch_add,
                                 initial_epoch=self.n_epochs,
                                 callbacks=[tboard_callback,
                                            model_checkpoint, earlyStopping],
                                 workers=1,
                                 )
        self.n_epochs = self.n_epochs + epoch_add
        self.model = model_forecasting_v2(input_shape=self.input_shape, n_outputs=len(
            self.productos), saved_file=self.checkpoint_path, summary=True)

    def readDatafromDB(self):
        productos = self.productos
        # Abriendo el archivo
        df_study = pd.read_excel(self.db_path, sheet_name=self.sheetdb)
        col_interes = self.datos_d_interes
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
        self.fechas_col = df_study_mod["FECHA"]  # Fechas
        # Pivotear las columnas para obtener las series de tiempo
        df_study_mod = df_study_mod.pivot(
            index="FECHA", columns=["PRODUCTO"], values=["DEMANDA_VENTAS"])
        df_study_mod = df_study_mod.sort_values(by="FECHA")
        df_study_mod.columns = df_study_mod.columns.droplevel()
        self.productos = list(df_study_mod.columns)

        return df_study_mod

    def gen_input(self, data=None, TRAIN=True):
        if TRAIN:
            data = self.db_clean.copy()
        window_agg = self.window_agg
        n_steps = self.n_steps
        productos = self.productos

        # ajustando el scaler para los datos que se serán de salida
        # scaler_y.fit(data)

        # Generando los datos agregados: Media movil y la STD movil
        data = add_roll_agg(data, Wmean=window_agg,
                            Wstd=window_agg, dropna=True)
        Y_columns = list(data.columns)

        if not TRAIN:  # Cuando se busca predecir y no entrenar
            # Generando los delays de tiempo
            # 11 delays + la actual posicion = 12
            data = gen_time_series(data, steps=n_steps-1, dropna=True)
            data = data.reset_index(drop=True)

            # Escalando los datos
            X = data
            X = self.scaler_x.transform(X)

            x_input = X[-1]
            x_input = np.expand_dims(x_input, axis=0)

            # Reshape vector
            # len(productos) * 3(valor, mean, std)
            n_features = int(len(productos)*3)
            x_input = x_input.reshape((-1, n_steps, n_features))
            input_shape = x_input.shape[1:]

            return x_input, input_shape, self.scaler_x, self.scaler_y

        else:
            scaler_y = MinMaxScaler()
            scaler_x = MinMaxScaler()

            data = gen_time_series(data, steps=n_steps, dropna=True)
            data = data.reset_index(drop=True)

            all_columns = list(data.columns)
            _ = [all_columns.remove(col) for col in Y_columns]

            Y_data = data[productos]
            scaler_y.fit(Y_data)
            Y_data = scaler_y.transform(Y_data)

            X = data[all_columns]
            scaler_x.fit(X)
            X = scaler_x.transform(X)
            # len(productos) * 3(valor, mean, std)
            n_features = int(len(productos)*3)
            X_data = X.reshape((-1, n_steps, n_features))

            # Periodos para estratificar
            bins = self.stratif
            periodos = pd.DataFrame(data.index)
            periodos["Periodos"] = pd.cut(periodos[0], bins=bins, labels=[
                                          "period"+str(i) for i in range(bins)])

            return X_data, Y_data, scaler_x, scaler_y, periodos

    def forecast_product(self, product_name=None, n_ahead=1, save_path=None):
        df_temp_accum = self.db_clean
        # Forecasting n ahead
        for _ in range(n_ahead):
            # Manipulamos la tabla total, transformamos y extraemos el ultimo dato
            x_input, input_shape, scaler_x, scaler_y = self.gen_input(
                data=df_temp_accum, TRAIN=False)

            # Forecasting
            forescast = self.model.predict(x_input)
            forescast_scal = scaler_y.inverse_transform(forescast)

            # Dataframe Forecating
            forescast_df = pd.DataFrame(forescast_scal, columns=self.productos)
            df_temp_accum = df_temp_accum.append(
                forescast_df, ignore_index=True)

        # Entregando producto en particular
        if product_name:
            df_predic = pd.DataFrame(
                df_temp_accum.iloc[-n_ahead:, self.productos.index(product_name)]).round()
            if save_path:
                df_predic.to_csv(path_or_buf=save_path)
            return df_predic

        if save_path:
            df_temp_accum.iloc[-n_ahead:].round().to_csv(path_or_buf=save_path)
        return df_temp_accum.iloc[-n_ahead:].round()

    def forecast_valdata(self, source="train"):
        if source == "val":
            forecast_scal = self.model.predict(self.X_test)
            forecast_scal = self.scaler_y.inverse_transform(forecast_scal)
            true_val = self.scaler_y.inverse_transform(self.y_test)
        elif source == "train":
            forecast_scal = self.model.predict(self.X_train)
            forecast_scal = self.scaler_y.inverse_transform(forecast_scal)
            true_val = self.scaler_y.inverse_transform(self.y_train)
        elif source == "whole":
            forecast_scal = self.model.predict(self.X_data)
            forecast_scal = self.scaler_y.inverse_transform(forecast_scal)
            true_val = self.scaler_y.inverse_transform(self.Y_data)

        forecast_scal = pd.DataFrame(
            forecast_scal, columns=self.productos).round()
        true_val = pd.DataFrame(true_val, columns=self.productos).round()

        return forecast_scal, true_val

    def forecasting_selfdata(self,):
        forecast_scal = self.model.predict(self.X_data)
        forecast_scal = self.scaler_y.inverse_transform(forecast_scal)
        true_val = self.scaler_y.inverse_transform(self.Y_data)

        forecast_scal = pd.DataFrame(
            forecast_scal, columns=self.productos).round()
        forecast_scal["Fecha"] = self.db_clean.index[-len(forecast_scal):]
        true_val = pd.DataFrame(true_val, columns=self.productos).round()
        true_val["Fecha"] = self.db_clean.index[-len(forecast_scal):]

        return forecast_scal, true_val


# %%

class Bulk_Models(object):
    def __init__(self, **kwargs):
        self.productos = kwargs["productos"]
        self.models = {}
        kwargs.pop("productos")
        _ = [self.models.update({producto: forecasting_demanda(
            **self.set_product(kwargs, producto))}) for producto in self.productos]

    @staticmethod
    def set_product(kwargs, producto):
        kwargs.update({"producto": producto})
        kwargs.update({"name_model": "model_"+producto})
        return kwargs

    def train_models(self, epochs=5000):
        for producto in self.productos:
            self.models[producto].train_model(epochs=epochs)
