import streamlit as st
import yfinance as yt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects  as go
from datetime import date

START="2015-01-01"

TODAY=date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction app")
stocks=("AAPL","GOOG","MSFT","GME")
selected_stocks=st.selectbox("Select dataset for prediction",stocks)
n_years=st.slider("Years of prediction",1,4)
period=n_years*365

@st.cache_data 
def load_data(ticker):
    data=yt.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data ...")
data=load_data(selected_stocks)
data_load_state.text("Loading data ...done!")

st.subheader("Raw Data")
st.write(data.head())


def  plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stocks close'))
    fig.layout.update(title_text='Time series analysis',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

