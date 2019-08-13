#importing necessary libraries
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

#Uploading the file
filename = 'BTC-USD.csv'
df = pd.read_csv(filename)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.head()

df.set_axis(df['Date'], inplace = True) #set index into date
df.drop(columns=['Open','High','Low','Volume'], inplace = True)
df.head()

!pip install plotly==3.6.0
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
print("Plotly Version: ",plotly.__version__)

init_notebook_mode(connected=True)
def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))
        

configure_plotly_browser_state()
trace = go.Scatter(x=df['Date'], 
                   y=df['Close'],
                   mode='lines', 
                   name='Data')
layout = go.Layout(title = "",
                   xaxis = {'title' : 'Date'},
                   yaxis = {'title' : 'Close(Dollars)'}
                  )
fig = go.Figure(data = [trace] , layout = layout)
iplot(fig)


close_data = df['Close'].values
close_data = close_data.reshape((-1,1))     #changing into 2d array

split_percent = 0.80                         #split data
split = int(split_percent * len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test =df['Date'][split:]

print(len(close_train))
print(len(close_test))


look_back = 15

train_generator = TimeseriesGenerator(close_train,
                                      close_train,
                                      length = look_back,
                                      batch_size = 25)
test_Generator = TimeseriesGenerator(close_test,
                                      close_test,
                                      length = look_back,
                                      batch_size = 1)
                                      
                                      
                                      
from keras.models import Sequential
from keras.layers import LSTM,Dense

model = Sequential()
model.add(
    LSTM(15,
        activation = 'relu',
        input_shape = (look_back, 1))

)
model.add(Dense(1))
model.compile(optimizer = 'adam' , loss = 'mse')

model.fit_generator(train_generator,epochs=100,verbose=1)



prediction = model.predict_generator(test_Generator)
prediction

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))


configure_






plotly_browser_state()
trace1 = go.Scatter(x = date_train,
                    y = close_train,
                    mode = 'lines',
                    name = 'Data')
trace2 = go.Scatter(x = date_test,
                    y = prediction,
                    mode = 'lines',
                    name = 'Prediction')
trace3 = go.Scatter(x = date_test,
                    y = close_test,
                    mode = 'lines',
                    name = 'Ground Truth')

layout = go.Layout(title = "BTS-USD",
                   xaxis = {'title' : 'Date'},
                   yaxis = {'title' : 'Close'}
                  )
fig = go.Figure(data = [trace1,trace2,trace3] , layout = layout)
iplot(fig) 
