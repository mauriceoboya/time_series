import os
os.chdir("/home/fibonacci/projects/time_series/")

import pandas as pd
import statsmodels
import streamlit as st
import yfinance as yt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects  as go
from datetime import date
from datetime import datetime
from statsmodels.tsa.stattools import adfuller




st.title('Kenya Inflation Rate Prediction')
n_years=st.slider("Years of prediction",1,4)
period=n_years*365


def load_data(nrows):
    dataset = pd.read_csv('InflationRates.csv', nrows=nrows)
    dataset['year'] = dataset.apply(lambda row: f"{row['Year']}-{row['Month']}", axis=1)
    dataset = dataset.drop(columns=['Month', 'Year'], axis=1)

    month_mapping = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }

    datetime_objects = []
    for date_string in dataset['year']:
        year, month_name = date_string.split('-')
        month = month_mapping[month_name]
        datetime_obj = date(int(year), month, 1)
        datetime_objects.append(datetime_obj) 
    dataset['year'] = datetime_objects 
    return dataset


load_state_data=st.write('Loading.....')
dataset=load_data(50)
st.write(dataset)



def  plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dataset.index,y=dataset['Annual Average Inflation'],name='Annual Inflation'))
    fig.add_trace(go.Scatter(x=dataset.index,y=dataset['12-Month Inflation'],name='Monthly Inflation'))
    fig.layout.update(title_text='Inflation Rates',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

dataset=dataset.drop(columns=['12-Month Inflation'],axis=1)
dataset.set_index('year', inplace=True)
dataset['Date'] = dataset.index


#adft = adfuller(dataset,autolag="AIC")
#output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", 
 #                                                       "critical value (1%)", #"critical value (5%)", "critical value (10%)"]})
#st.write(output_df)


df_train=dataset[["Date",'Annual Average Inflation']]
df_train=df_train.rename(columns={"Date":"ds","Annual Average Inflation":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader("Forecast Inflation data")
st.write(forecast.tail())

st.subheader('Forecasted Inflation')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('Forecasted components')
fig2=m.plot_components(forecast)
st.write(fig2)