"""Global BLOC functional Streamlit dashboard."""
import math
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from global_bloc_finance.visualization_registry import visualization_options

st.set_page_config(page_title="Global BLOC", page_icon="🌐", layout="wide")
st.title("Global BLOC Financial Intelligence")
st.caption("Live data • executable research modules • 350-metric architecture")

@st.cache_data(ttl=900)
def market_data(ticker, period="2y", interval="1d"):
    d=yf.download(ticker,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(how="all")

@st.cache_data(ttl=900)
def info(ticker): return yf.Ticker(ticker).info

@st.cache_data(ttl=3600)
def worldbank(indicator,country,start=2000,end=None):
    end=end or date.today().year
    u=f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    r=requests.get(u,params={"format":"json","per_page":1000,"date":f"{start}:{end}"},timeout=20)
    r.raise_for_status()
    return pd.DataFrame([{"year":int(x["date"]),"value":x["value"]} for x in r.json()[1] if x["value"] is not None]).sort_values("year")

@st.cache_data(ttl=3600)
def sec_filings(cik):
    u=f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    r=requests.get(u,headers={"User-Agent":"GlobalBLOC research dashboard contact@example.com"},timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json()["filings"]["recent"])

def overview():
    st.header("Overview"); ticker=st.text_input("Ticker","AAPL").upper().strip(); period=st.selectbox("History",["6mo","1y","2y","5y","10y","max"],index=2)
    if st.button("Load live market data",type="primary"):
        d=market_data(ticker,period)
        if d.empty: st.error("No market data returned.")
        else:
            close=d.Close.dropna(); ret=close.pct_change().dropna(); a,b,c=st.columns(3)
            a.metric("Last Close",f"{close.iloc[-1]:,.2f}"); b.metric("Total Return",f"{(close.iloc[-1]/close.iloc[0]-1)*100:.2f}%"); c.metric("Annualized Volatility",f"{ret.std()*np.sqrt(252)*100:.2f}%")
            st.plotly_chart(px.line(d,y="Close",title=f"{ticker} Price History"),use_container_width=True)
            m=pd.DataFrame({"Metric":["Return","Volatility","Sharpe","Max Drawdown"],"Value":[close.iloc[-1]/close.iloc[0]-1,ret.std()*np.sqrt(252),ret.mean()/ret.std()*np.sqrt(252),(close/close.cummax()-1).min()]})
            st.dataframe(m,use_container_width=True,hide_index=True); st.download_button("Download CSV",d.to_csv().encode(),f"{ticker}_market.csv","text/csv")

def economics():
    st.header("Economics"); countries={"United States":"USA","China":"CHN","Germany":"DEU","Japan":"JPN","United Kingdom":"GBR","India":"IND","Canada":"CAN","Brazil":"BRA"}; inds={"GDP growth":"NY.GDP.MKTP.KD.ZG","Inflation":"FP.CPI.TOTL.ZG","Unemployment":"SL.UEM.TOTL.ZS","Exports (% GDP)":"NE.EXP.GNFS.ZS","Imports (% GDP)":"NE.IMP.GNFS.ZS"}
    c=st.selectbox("Country",list(countries)); name=st.selectbox("Indicator",list(inds))
    if st.button("Load economic data",type="primary"):
        try:
            d=worldbank(inds[name],countries[c]); st.plotly_chart(px.line(d,x="year",y="value",markers=True,title=f"{c}: {name}"),use_container_width=True); st.dataframe(d.sort_values("year",ascending=False),use_container_width=True,hide_index=True); st.download_button("Download economic CSV",d.to_csv(index=False).encode(),f"{countries[c]}_economic.csv","text/csv")
        except Exception as e: st.error(f"Request failed: {e}")

def exchanges():
    st.header("Stock Exchanges"); symbols=st.text_input("Tickers, comma-separated","AAPL,MSFT,NVDA,TSM").upper()
    if st.button("Load exchange data",type="primary"):
        rows=[]
        for s in [x.strip() for x in symbols.split(",") if x.strip()]:
            try:
                x=info(s); rows.append({"Ticker":s,"Company":x.get("longName",""),"Exchange":x.get("exchange",""),"Market":x.get("market",""),"Country":x.get("country",""),"Currency":x.get("currency",""),"Sector":x.get("sector","")})
            except Exception as e: rows.append({"Ticker":s,"Error":str(e)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

def regulation():
    st.header("Banking & Regulation"); ticker=st.text_input("SEC company ticker","JPM").upper().strip()
    if st.button("Load SEC filings",type="primary"):
        try:
            x=info(ticker); cik=x.get("cik")
            if not cik: st.warning("No SEC CIK returned for this ticker.")
            else:
                d=sec_filings(cik); forms=st.multiselect("Forms",sorted(d.form.dropna().unique()),default=[x for x in ["10-K","10-Q","8-K"] if x in set(d.form)])
                view=d[d.form.isin(forms)][["accessionNumber","filingDate","reportDate","form","primaryDocument","primaryDocDescription"]].head(100)
                st.dataframe(view,use_container_width=True,hide_index=True); st.download_button("Download SEC filing metadata",view.to_csv(index=False).encode(),f"{ticker}_SEC.csv","text/csv"); st.link_button("Open official SEC filing search",f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}")
        except Exception as e: st.error(f"SEC request failed: {e}")

def trade():
    st.header("Import / Export"); countries={"United States":"USA","China":"CHN","Germany":"DEU","Japan":"JPN","United Kingdom":"GBR","India":"IND","Canada":"CAN"}; c=st.selectbox("Country",list(countries)); measure=st.radio("Measure",["Exports (% GDP)","Imports (% GDP)"],horizontal=True); code="NE.EXP.GNFS.ZS" if measure.startswith("Exports") else "NE.IMP.GNFS.ZS"
    if st.button("Load trade data",type="primary"):
        try:
            d=worldbank(code,countries[c]); st.plotly_chart(px.bar(d,x="year",y="value",title=f"{c}: {measure}"),use_container_width=True); st.dataframe(d,use_container_width=True,hide_index=True)
        except Exception as e: st.error(f"Request failed: {e}")

def sectors():
    st.header("Industry Sectors"); ticker=st.text_input("Company ticker","XOM").upper().strip()
    if st.button("Load sector data",type="primary"):
        x=info(ticker); a,b,c=st.columns(3); a.metric("Sector",x.get("sector","—")); b.metric("Industry",x.get("industry","—")); c.metric("Market Cap",str(x.get("marketCap","—"))); st.json({k:x.get(k) for k in ["longName","sector","industry","country","exchange","fullTimeEmployees","revenueGrowth","profitMargins"]})

def ml():
    st.header("ML / Deep Learning"); ticker=st.text_input("Training ticker","AAPL").upper().strip(); horizon=st.selectbox("Target horizon (trading days)",[1,5,21]); model_name=st.selectbox("Model",["Linear Regression","Random Forest"])
    if st.button("Train model",type="primary"):
        d=market_data(ticker,"5y"); r=d.Close.pct_change().dropna(); z=pd.DataFrame({"target":r.shift(-horizon),"lag1":r.shift(1),"lag2":r.shift(2),"lag5":r.shift(5),"vol20":r.rolling(20).std()}).dropna(); X=z[["lag1","lag2","lag5","vol20"]]; y=z.target; split=int(len(z)*.8); Xtr,Xte,ytr,yte=X.iloc[:split],X.iloc[split:],y.iloc[:split],y.iloc[split:]; model=LinearRegression() if model_name=="Linear Regression" else RandomForestRegressor(n_estimators=200,random_state=42); model.fit(Xtr,ytr); p=model.predict(Xte); a,b,c=st.columns(3); a.metric("MAE",f"{mean_absolute_error(yte,p):.6f}"); b.metric("RMSE",f"{math.sqrt(mean_squared_error(yte,p)):.6f}"); c.metric("R²",f"{r2_score(yte,p):.4f}"); out=pd.DataFrame({"Actual":yte,"Predicted":p},index=yte.index); st.plotly_chart(px.line(out,title=f"{ticker}: actual vs predicted return"),use_container_width=True); st.dataframe(out.tail(50),use_container_width=True)

def explorer():
    st.header("Data Explorer"); ticker=st.text_input("Ticker","AAPL").upper().strip(); period=st.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],index=3)
    if st.button("Query live dataset",type="primary"):
        d=market_data(ticker,period); st.dataframe(d,use_container_width=True); st.download_button("Export CSV",d.to_csv().encode(),f"{ticker}_{period}.csv","text/csv"); st.download_button("Export JSON",d.reset_index().to_json(orient="records").encode(),f"{ticker}_{period}.json","application/json")

def installation():
    st.header("Installation"); packages=["streamlit","pandas","numpy","plotly","scikit-learn","statsmodels","xgboost","yfinance","requests"]; rows=[]
    for p in packages:
        try: m=__import__(p.replace("-","_")); rows.append({"Package":p,"Status":"Installed","Version":getattr(m,"__version__","unknown")})
        except Exception as e: rows.append({"Package":p,"Status":"Missing","Version":str(e)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True); st.code("pip install -r requirements.txt",language="bash"); st.info("Runtime diagnostics only; no silent OS changes.")

def metric_visualizations():
    st.header("350 Metrics — Visualization Workspace"); metric={"id":1,"name":"Simple Return","category":"return"}; opts=visualization_options(metric); chosen=st.selectbox("Visualization",opts,format_func=lambda x: ("* " if x["recommended"] else "")+x["name"]); st.caption(chosen["reason"]); st.write("Recommendations identify useful starting points; they do not restrict selection.")

pages={"Overview":overview,"Economics":economics,"Stock Exchanges":exchanges,"Banking & Regulation":regulation,"Import / Export":trade,"Industry Sectors":sectors,"ML / Deep Learning":ml,"Data Explorer":explorer,"Installation":installation,"350 Metrics":metric_visualizations}
selection=st.sidebar.radio("Global BLOC modules",list(pages)); pages[selection]()
