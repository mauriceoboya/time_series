import streamlit as st
import yfinance as yt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects  as go
from datetime import date

START="2015-01-01"

TODAY=date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction")
stocks=("MANU","Nasdaq","MSFT","GME")
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

##forecasting

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)


st.subheader("Forecast Data")
st.write(forecast.tail())


st.write('Forecasted stocks')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('Forecasted components')
fig2=m.plot_components(forecast)
st.write(fig2)