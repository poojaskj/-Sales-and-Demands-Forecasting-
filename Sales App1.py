import streamlit as st
import numpy as np
import pandas as pd
import altair as alt 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import statsmodels
import sqlite3
import plotly.graph_objects as go

st.title('Sales Predictor')
st.markdown("   ")
st.markdown("   ")
st.subheader("WELCOME")
st.write(" Sales Predictor- an app used to predict the future values from the time series Data. The Model Chosed is given the best asccuracies i.e. over90 %. ")
st.write("We would Love to help you by predicting future sales of your Product") 
st.markdown("   ")
st.markdown("   ")
st.subheader('Update /Upload File')

st.markdown("----")
    
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "sql"])

if uploaded_file is not None:
    
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "sql":
        query = st.text_area("Enter the SQL query")
        if st.button("Execute Query"):
            conn =sqlite3.connect(uploaded_file)
            df = pd.read_sql_query(query, conn)
    else:
        df = pd.read_excel(uploaded_file) if file_extension == "xlsx" else pd.read_csv(uploaded_file)
    column_list = df.columns.tolist()
    
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')

    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.write(df.head(10))
    
    st.write("Shape of Data: " , df.shape)

    st.markdown("----")

    # Sidebar
    st.sidebar.header("Time Series Forecasting")
    st.sidebar.write("A time series is a sequence where a metric is recorded over regular time intervals. Forecasting is the next step where you want to predict the future values the series is going to take.")
    st.sidebar.markdown("----")
    st.sidebar.header("Model Used - ARIMA  (‘AutoRegressive Integrated Moving Average)")
    st.sidebar.write(" It is a forecasting algorithm based on the information from the historical data of the time series can alone be used to predict the future values. An ARIMA model is characterized by 3 terms: p, d, q where, p is the order of the AR term, q is the order of the MA term, d is the number of differencing required to make the time series stationary.")
    st.sidebar.markdown("----")
    st.sidebar.write("Here We Take the p,d q order as 6,1,6.")
    st.sidebar.write("The Data is Seasonal so model has converted to SARIMAX i.e. Seasonal-ARIMA")
    st.sidebar.write("If the data is not stationary we have converted it to stationary using ADF test")
    st.sidebar.write("Also we have implemented Accuracy metrics")

    st.subheader("Exploratory Data Analysis")

    with st.expander("Descriptive Summary"):
        st.dataframe(df.describe())

    st.markdown("----")

    st.subheader("Plotting")

    with st.expander("Altair Chart"):
        chart = alt.Chart(df).mark_circle().encode(
            x='Month',
            y='Sales_MT'
            ).interactive()
       
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

    st.markdown("----")
    
    with st.expander("Line Chart"):
        st.line_chart(data=df, y= ['Sales_MT'])

    st.markdown("----")


       # Model
    # Outlier treatment With Winsorization
    from feature_engine.outliers import Winsorizer
    
    winsor = Winsorizer(capping_method ='iqr',
                       fold =1.5,
                       tail='both',
                       variables=['Sales_MT'])
    
    df = winsor.fit_transform(df)
    
    df = df.set_index("Month")
    
    Dfs = df.iloc[:,1]
    
    
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    # Decomposing Sales Data / TimeSeries data into Components level,trend ,sesonal,noise
    decompose_ts_add = seasonal_decompose(Dfs, model = "", period = 3)
    print(decompose_ts_add.trend)
    print(decompose_ts_add.seasonal)
    print(decompose_ts_add.resid)
    print(decompose_ts_add.observed)
    
    st.write("TimeSeries data into Components level,trend ,sesonal,noise: ",decompose_ts_add.plot())
    
    st.markdown("----")
    
    st.header("Let's Forecast")
    
    period = st.slider("Months of Predictions: ", 0,12)
    
    
    from statsmodels.tsa.stattools import adfuller
    
    #perform augmented Dickey-Fuller test
    adfuller(Dfs)

    ##### ACF Plots Visualization For Sales Data
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    plt.rcParams.update({'figure.figsize':(25,7), 'figure.dpi':120})
    
    
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(Dfs); axes[0, 0].set_title('Original Series')
    plot_acf(Dfs, ax=axes[0, 1])
    
    # 1st Differencing
    axes[1, 0].plot(Dfs.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(Dfs.diff().dropna(), ax=axes[1, 1])
    
    # 2nd Differencing
    axes[2, 0].plot(Dfs.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(Dfs.diff().diff().dropna(), ax=axes[2, 1])
    
    plt.show()
    
    # PACF plot of 1st differenced series
    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(Dfs.diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))
    plot_pacf(Dfs.diff().dropna(), ax=axes[1])
    
    plt.show()
    
    ### Splitting data into training And testing dataset
    train_size = int(len(Dfs)*0.7)
    test_size = len(Dfs)-train_size
    
    
    
    # Create Training and Test Data
    train = Dfs[:84]
    test = Dfs[84:]
    
    # Build Model
    
    from statsmodels.tsa.arima.model import ARIMA
    
    # model = ARIMA(train, order=(6,1,6))  
    model_A= ARIMA(train, order=(6, 1, 6))  
    fitted = model_A.fit()  
    start_index = len(df)
    end_index = start_index + period
    forecast = fitted.predict(start=start_index, end=end_index)
    
    st.markdown("   ")

    if st.button("Predict", use_container_width=True)==0:
      
         if (period==0):
             st.write("Please Select The Time Period You want to Forecast For & Click on Predict button")
         else : 
             st.write("You have selected ", period,"Months, Now Click On Predict")
            
    else:
        st.success("Predictions Ready!......")
        st.subheader("Forecasted Sales Values")
        st.write(forecast.tail(period))
        st.subheader("Accuracy metrics")
        def forecast_accuracy(forecast, actual):
            mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
            me = np.mean(forecast - actual)             # ME
            mae = np.mean(np.abs(forecast - actual))    # MAE
            mpe = np.mean((forecast - actual)/actual)   # MPE
            rmse = np.mean((forecast - actual)**2)**.5  # RMSE
            corr = np.corrcoef(forecast, actual)[0,1]   # corr
            mins = np.amin(np.hstack([forecast[:,None], 
                                  actual[:,None]]), axis=1)
            maxs = np.amax(np.hstack([forecast[:,None], 
                                  actual[:,None]]), axis=1)
            minmax = 1 - np.mean(mins/maxs)             # minmax
            return({'mape':mape, 'me':me, 'mae': mae, 
                    'mpe': mpe, 'rmse':rmse,  
                    'corr':corr, 'minmax':minmax})
      
        A = forecast_accuracy(forecast[:period], test.values[:period])
        st.write(A)
        
        Accuracy = 100-( np.mean(np.abs(forecast[:period] - test.values[:(period)])/np.abs(test.values[:(period)])))*100
       
        st.metric(label="Accuracy of model", value= Accuracy)
       
        
        st.markdown("----")
       
        st.title("Actual values vs Forecasted values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df.Sales_MT, name='Actual'))
        fig.add_trace(go.Scatter(x= forecast.index, y=forecast, name='Predicted'))
        fig.update_layout(title="Time Series Plot")
        st.plotly_chart(fig)
        
        
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")

st.subheader("Thanking You...")
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")
st.markdown("   ")
col1, col2 = st.columns(2)


col1.write("email - poojaskj29@gmail.com")

col2.write("Linkdin - linkedin.com/in/pooja-suryawanshi-979b06197")
col2.write("Github - https://github.com/poojaskj/-Sales-and-Demands-Forecasting-")


