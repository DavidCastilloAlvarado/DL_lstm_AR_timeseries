# Forecasting Deep Learning Model

This work is inspired in [LSTNet](https://github.com/fbadine/LSTNet) from 2018. If you want to check the original model, check that up. In this implementation I’d used an autoregressive model in parallel with one cell lstm model.

This work is an adaptation of LSTNet to work with time series for forecasting predictions, this technique provides two effects, the ARmodel provide capability to short term time correlation while the LSTM model help with long term and correlations and no linearities behaviors.

Arquitecture:

Four custom layers were created to preprocesing and create the model.

```python
y1 = prelstm(x)
y1 = LSTMbulk(y1)
y2 = transAR(x)
y2 = ARbulk(y2)
out = addLayer([y1, y2])
model = Model(inputs=x, outputs=out)
```

 <center>
    <img src="img/red.PNG" alt="drawing" width="700"/>
</center>
