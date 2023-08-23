import pandas as pd
import streamlit as st
import yfinance as yt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects  as go
from datetime import date
from datetime import datetime


st.write('Kenya Inflation Rate Prediction')
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
    dataset.index = dataset['year']
    del dataset['year']
    return dataset


load_state_data=st.write('Loading.....')
dataset=load_data(50)
st.write(dataset)

st.write(dataset.columns)

def  plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=['year'],y=dataset['Annual Average Inflation'],name='Annual Inflation'))
    fig.add_trace(go.Scatter(x=['year'],y=dataset['12-Month Inflation'],name='Monthly Inflation'))
    fig.layout.update(title_text='Inflation Rates',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


df_train=dataset[["year",'Annual Average Inflation']]
df_train=df_train.rename(columns={"year":"ds","Annual Average Inflation":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)