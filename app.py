"""Global BLOC functional Streamlit dashboard."""
import json
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
from global_bloc_finance.economic_concepts import MACRO_CONCEPTS, MICRO_CONCEPTS
from global_bloc_finance.investment_metrics import METRICS, calculate_metrics, metric_catalog
from global_bloc_finance.recession_intelligence import assess_country, assess_bloc
from global_bloc_finance.research_providers import (provider_dataframe, priority_provider_dataframe, regulatory_dataframe, REGULATORY_SOURCE_CONFIG, RATING_API_TEMPLATES, product_dataframe, credentialed_request_template)
from global_bloc_finance.concept_securities import securities_for_concept, concept_security_categories
from global_bloc_finance.api_ingestion import APIConfig, AUTH_METHODS, request_api, normalize_ingested_data, analytical_summary, trend_summary
# Executable Business Cycle concept application
CYCLE_INDICATORS = {"Real GDP Growth":"NY.GDP.MKTP.KD.ZG","Inflation":"FP.CPI.TOTL.ZG","Unemployment":"SL.UEM.TOTL.ZS","Investment Growth":"NE.GDI.FTOT.KD.ZG"}
COUNTRY_ISO3 = {"United States":"USA","China":"CHN","Germany":"DEU","Japan":"JPN","United Kingdom":"GBR","India":"IND","Canada":"CAN","Brazil":"BRA","Australia":"AUS","South Korea":"KOR","Mexico":"MEX","France":"FRA","Italy":"ITA","Spain":"ESP","Singapore":"SGP","Saudi Arabia":"SAU","United Arab Emirates":"ARE"}

def cycle_series(iso3, start=1990):
    frames=[]
    for name,code in CYCLE_INDICATORS.items():
        try:
            d=worldbank_country(code,iso3,start=start).rename(columns={"value":name})
            if not d.empty: frames.append(d.set_index("year"))
        except Exception:
            pass
    if not frames: return pd.DataFrame()
    d=pd.concat(frames,axis=1).sort_index().reset_index()
    d["GDP Growth Change"]=d["Real GDP Growth"].diff()
    d["Unemployment Change"]=d["Unemployment"].diff()
    d["Inflation Change"]=d["Inflation"].diff()
    return d

def _cycle_slope(series, window=3):
    s=pd.Series(series).dropna().tail(window)
    return float(np.polyfit(np.arange(len(s)),s.to_numpy(),1)[0]) if len(s)>=2 else np.nan

def classify_business_cycle(d):
    if d.empty or "Real GDP Growth" not in d or len(d.dropna(subset=["Real GDP Growth"]))<5:
        return {"phase":"Insufficient Data","confidence":0.0,"score":0.0}
    d=d.dropna(subset=["Real GDP Growth"]); g=d["Real GDP Growth"]; u=d["Unemployment"]
    latest=float(g.iloc[-1]); slope=_cycle_slope(g); uslope=_cycle_slope(u); accel=float(g.iloc[-1]-g.iloc[-2])
    score=2 if latest>2 else 1 if latest>0 else -2 if latest<-1 else -1
    score += 1 if slope>.25 else -1 if slope<-.25 else 0
    score += 1 if pd.notna(uslope) and uslope<-.15 else -1 if pd.notna(uslope) and uslope>.15 else 0
    score += .5 if accel>.4 else -.5 if accel<-.4 else 0
    recent=g.tail(4); peak=len(recent)>=3 and g.iloc[-1]<recent.max() and slope<0; trough=len(recent)>=3 and g.iloc[-1]>recent.min() and slope>0
    if score>=2: phase="Trough" if trough else "Expansion"
    elif score<=-2: phase="Peak" if peak else "Contraction"
    elif peak: phase="Peak"
    elif trough: phase="Trough"
    else: phase="Expansion" if slope>=0 else "Contraction"
    evidence=np.mean([latest>0,slope>0,pd.notna(uslope) and uslope<0,accel>0])
    confidence=evidence if phase in {"Expansion","Trough"} else 1-evidence
    return {"phase":phase,"confidence":round(float(max(0,min(1,confidence))),2),"score":round(float(score),2),"gdp_growth":latest,"gdp_slope":slope,"unemployment_slope":uslope,"inflation":float(d["Inflation"].iloc[-1]) if pd.notna(d["Inflation"].iloc[-1]) else np.nan}

def business_cycle_panel(iso3,label):
    d=cycle_series(iso3)
    if d.empty:
        st.warning(f"No sufficient macroeconomic data returned for {label} ({iso3})."); return
    current=classify_business_cycle(d); phase=current["phase"]
    st.subheader(f"Business Cycle — {label}")
    st.caption("Applied concept: Business Cycle. The classification uses GDP growth, GDP momentum, unemployment direction and inflation context. It is an analytical classification, not an official recession declaration.")
    a,b,c,dcol=st.columns(4)
    a.metric("Current Phase",phase); b.metric("Confidence",f"{current['confidence']*100:.0f}%"); c.metric("Real GDP Growth",f"{current.get('gdp_growth',np.nan):.2f}%"); dcol.metric("Inflation",f"{current.get('inflation',np.nan):.2f}%")
    view=d.copy()
    view["Phase"]=[classify_business_cycle(view.iloc[:i+1])["phase"] for i in range(len(view))]
    st.plotly_chart(px.line(view,x="year",y="Real GDP Growth",markers=True,title=f"{label}: Real GDP Growth with Business-Cycle Context"),use_container_width=True)
    st.dataframe(view[["year","Real GDP Growth","Inflation","Unemployment","Investment Growth","Phase"]].tail(20),use_container_width=True,hide_index=True)
    st.info(f"Concept application: {phase} is the current analytical regime. GDP slope = {current.get('gdp_slope',np.nan):.2f}; unemployment slope = {current.get('unemployment_slope',np.nan):.2f}. These are evidence for analysis, not a guaranteed forecast.")

from global_bloc_finance.global_intelligence import (
    FINANCIAL_DOMAINS, ECONOMIC_BLOCS, REGIONS, MAP_INDICATORS,
    worldbank_all, worldbank_country, bloc_members, map_figure,
    bloc_market_summary
)

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


def recession_intelligence_panel(iso3, label):
    with st.spinner(f"Analyzing {label} macroeconomic and country-level activity..."):
        assessment = assess_country(iso3, label)
    if assessment.get("status") == "Insufficient data":
        st.warning(f"Insufficient data for {label}.")
        return
    st.subheader(f"Recession & Business-Cycle Assessment — {label}")
    st.caption("Transparent analytical screen using annual macroeconomic and activity indicators. It is not an official recession dating decision.")
    a,b,c,d = st.columns(4)
    a.metric("Assessment", assessment["status"])
    b.metric("Confidence", f'{assessment["confidence"]*100:.0f}%')
    c.metric("Real GDP Growth", f'{assessment["real_gdp_growth"]:.2f}%')
    d.metric("Broad Weakening Signals", str(assessment["broad_weakening_signals"]))
    evidence = pd.DataFrame([
        ["Real GDP growth", assessment.get("real_gdp_growth"), "Macro"],
        ["GDP per-capita growth", assessment.get("gdp_per_capita_growth"), "Macro"],
        ["Inflation", assessment.get("inflation"), "Macro"],
        ["Unemployment trend", assessment.get("unemployment_slope"), "Macro / labor"],
        ["Household consumption growth", assessment.get("household_consumption_growth"), "Micro / household demand"],
        ["Investment growth", assessment.get("investment_growth"), "Micro / business investment"],
        ["Trade growth", assessment.get("trade_growth"), "External sector"],
    ], columns=["Indicator","Latest / trend","Analytical layer"])
    st.dataframe(evidence, use_container_width=True, hide_index=True)
    panel = assessment["panel"].reset_index()
    if "real_gdp_growth" in panel:
        st.plotly_chart(px.line(panel, x="year", y="real_gdp_growth", markers=True, title=f"{label}: Real GDP Growth"), use_container_width=True)
    sector_cols = [x for x in ["manufacturing_value_added_growth","industry_value_added_growth","services_value_added_growth","agriculture_value_added_growth"] if x in panel]
    if sector_cols:
        latest = panel.tail(1)
        sector_view = latest[["year"] + sector_cols].melt(id_vars="year", var_name="Sector metric", value_name="Growth")
        st.dataframe(sector_view, use_container_width=True, hide_index=True)
    st.info("A recession signal means the data satisfy GlobalBLOCS's transparent screening rule. It does not claim that a national statistical authority has officially dated a recession.")

def bloc_recession_panel(bloc, members):
    with st.spinner(f"Analyzing {bloc}..."):
        result = assess_bloc(members)
    st.subheader(f"Economic BLOC Recession Assessment — {bloc}")
    if result["table"].empty:
        st.warning("Insufficient data for this BLOC.")
        return
    a,b,c,d = st.columns(4)
    a.metric("BLOC Assessment", result["status"])
    b.metric("Countries with Recession Signal", f'{result["recession_share"]*100:.0f}%')
    c.metric("Contraction / Risk", f'{result["contraction_share"]*100:.0f}%')
    d.metric("Mean Real GDP Growth", f'{result["mean_real_gdp_growth"]:.2f}%')
    table = result["table"].copy()
    st.dataframe(table.sort_values(["status","real_gdp_growth"]), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(table.sort_values("real_gdp_growth"), x="iso3", y="real_gdp_growth", color="status", title=f"{bloc}: Real GDP Growth by Economy"), use_container_width=True)
    st.caption("BLOC classification uses member-country breadth plus average real GDP growth. It is an analytical BLOC-wide signal, not an official supranational recession declaration.")

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
            try:
                company=info(ticker); country=company.get("country"); iso3=COUNTRY_ISO3.get(country)
                if iso3: business_cycle_panel(iso3,f"{ticker} company economy ({country})")
                else: st.info(f"Business-cycle mapping is not yet available for company country: {country or 'Unknown'}")
            except Exception as e: st.warning(f"Company-economy cycle analysis unavailable: {e}")

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


US_MACRO_INDICATORS = {
    "GDP & Output": {
        "Real GDP":"GDPC1","Real GDP Growth":"A191RL1Q225SBEA","GDP Deflator":"GDPDEF",
        "Personal Consumption":"PCECC96","Private Investment":"GPDI","Government Spending":"GCEC1",
        "Net Exports":"NETEXP"
    },
    "Inflation": {
        "CPI":"CPIAUCSL","Core CPI":"CPILFESL","PCE":"PCEPI","Core PCE":"PCEPILFE",
        "PPI":"PPIACO","Import Prices":"IR3TIB01USM156N","Employment Cost Index":"ECIWAG"
    },
    "Labor Market": {
        "Unemployment Rate":"UNRATE","Nonfarm Payrolls":"PAYEMS","Labor Force Participation":"CIVPART",
        "Average Hourly Earnings":"CES0500000003","Job Openings":"JTSJOL",
        "Initial Claims":"ICSA","Continuing Claims":"CCSA","Quits":"JTSQUR"
    },
    "Consumer": {
        "Retail Sales":"RSAFS","Real Personal Income":"W875RX1","Personal Income":"PI",
        "Personal Saving Rate":"PSAVERT","Consumer Credit":"TOTALSL"
    },
    "Housing": {
        "Housing Starts":"HOUST","Building Permits":"PERMIT","New Home Sales":"HSN1F",
        "Existing Home Sales":"EXHOSLUSM495S","Case-Shiller Home Price Index":"CSUSHPINSA",
        "30Y Mortgage Rate":"MORTGAGE30US","Housing Inventory":"HNFSEPUSSA"
    },
    "Manufacturing": {
        "Industrial Production":"INDPRO","Capacity Utilization":"TCU",
        "Manufacturers New Orders":"AMTMNO","Durable Goods Orders":"DGORDER",
        "Business Inventories":"BUSINV","Labor Productivity":"OPHPBS"
    },
    "Financial Markets": {
        "S&P 500":"SP500","10Y Treasury Yield":"DGS10","2Y Treasury Yield":"DGS2",
        "10Y-2Y Spread":"T10Y2Y","Corporate Bond Spread":"BAA10Y",
        "VIX":"VIXCLS","Trade Weighted Dollar":"DTWEXBGS","Gold":"GOLDAMGBD228NLBM",
        "Crude Oil":"DCOILWTICO"
    },
    "Banking & Money": {
        "M2":"M2SL","Bank Credit":"H8B1247NCBCMG","Commercial & Industrial Loans":"BUSLOANS",
        "Fed Total Assets":"WALCL","Reserve Balances":"WRESBAL","Reverse Repo":"RRPONTSYD"
    },
    "Federal Reserve": {
        "Federal Funds Rate":"FEDFUNDS","Discount Rate":"DISCOUNT",
        "10Y Treasury Yield":"DGS10","Fed Total Assets":"WALCL",
        "Reserve Balances":"WRESBAL"
    },
    "Government & Fiscal": {
        "Federal Debt":"GFDEBTN","Federal Debt / GDP":"GFDEGDQ188S",
        "Federal Receipts":"FGRECPT","Federal Expenditures":"FGEXPND",
        "Federal Budget Balance":"MTSDS133FMS"
    },
    "International": {
        "Trade Balance":"BOPGSTB","Current Account":"IEABC",
        "Exports":"EXPGSC1","Imports":"IMPGSC1","Broad Dollar Index":"DTWEXBGS"
    },
    "Productivity & Innovation": {
        "Labor Productivity":"OPHPBS","Total Factor Productivity":"RTFPNAUSA632NRUG",
        "Private Fixed Investment":"PNFI","R&D Investment":"Y057RC1Q027SBEA"
    }
}

US_MACRO_PILLARS = {
    "Growth":"GDP, Industrial Production, Retail Sales, Housing, PMI/ISM",
    "Labor":"Unemployment, Payrolls, JOLTS, Claims, Wages",
    "Inflation":"CPI, PCE, PPI, Employment Costs, GDP Deflator",
    "Financial Conditions":"Fed Funds, Treasury Curve, Credit Spreads, Dollar, VIX, Money Supply"
}

@st.cache_data(ttl=900)
def fred_series(series_id, start="2000-01-01"):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id":series_id,"cosd":start}, timeout=20)
    r.raise_for_status()
    d = pd.read_csv(pd.io.common.BytesIO(r.content))
    value_col = [x for x in d.columns if x.upper() != "DATE"][0]
    d["DATE"] = pd.to_datetime(d["observation_date"] if "observation_date" in d.columns else d["DATE"], errors="coerce")
    d["value"] = pd.to_numeric(d[value_col], errors="coerce")
    return d[["DATE","value"]].dropna().sort_values("DATE")

def us_business_cycle_screen(data):
    if data.empty or len(data) < 8:
        return {"phase":"Insufficient Data","confidence":0.0}
    latest = data.iloc[-1]
    g = float(latest.get("Real GDP Growth", np.nan))
    u_slope = data["Unemployment Rate"].dropna().tail(6)
    u_slope = _cycle_slope(u_slope, min(6, len(u_slope))) if len(u_slope) >= 2 else np.nan
    ip_slope = _cycle_slope(data["Industrial Production"].dropna().tail(6), 6) if data["Industrial Production"].notna().sum() >= 2 else np.nan
    spread = float(latest.get("10Y-2Y Spread", np.nan))
    score = 0
    score += 2 if g > 2 else 1 if g > 0 else -2 if g < -1 else -1
    score += -1 if pd.notna(u_slope) and u_slope > 0.05 else 1 if pd.notna(u_slope) and u_slope < -0.05 else 0
    score += 1 if pd.notna(ip_slope) and ip_slope > 0 else -1 if pd.notna(ip_slope) and ip_slope < 0 else 0
    score += 1 if spread > 0 else -1
    if score >= 3: phase="Expansion"
    elif score <= -2: phase="Contraction"
    else: phase="Peak" if g > 2 and spread < 0 else "Trough" if g <= 0 and ip_slope > 0 else "Transition"
    return {"phase":phase,"score":score,"confidence":round(min(1,max(0,abs(score)/5)),2)}

def us_macro_dashboard():
    st.header("🇺🇸 U.S. Macroeconomic Analysis")
    st.caption("The U.S. economy is used as a living macroeconomic case study. Data are organized by business-cycle phase, economic sector and the four professional macro pillars.")
    st.info("This module is an analytical research screen. Business-cycle classifications are model-derived; official recession dating is represented separately through the NBER/FRED recession indicator when available.")

    st.subheader("Level 1 — Business Cycle")
    cycle_data = {}
    for label, sid in {
        "Real GDP Growth":"A191RL1Q225SBEA","Unemployment Rate":"UNRATE",
        "Industrial Production":"INDPRO","10Y-2Y Spread":"T10Y2Y"
    }.items():
        try:
            cycle_data[label] = fred_series(sid, "1990-01-01").set_index("DATE")["value"]
        except Exception:
            cycle_data[label] = pd.Series(dtype=float)
    cycle = pd.concat(cycle_data, axis=1).dropna(how="all")
    cycle_screen = us_business_cycle_screen(cycle)
    a,b,c,d = st.columns(4)
    a.metric("Analytical Phase", cycle_screen["phase"])
    b.metric("Screen Confidence", f'{cycle_screen["confidence"]*100:.0f}%')
    b2 = cycle["Real GDP Growth"].dropna()
    c.metric("Latest GDP Growth", "N/A" if b2.empty else f"{b2.iloc[-1]:.2f}%")
    d.metric("10Y−2Y Spread", "N/A" if cycle["10Y-2Y Spread"].dropna().empty else f'{cycle["10Y-2Y Spread"].dropna().iloc[-1]:.2f}%')

    try:
        rec = fred_series("USREC", "1960-01-01")
        st.subheader("Official recession reference")
        st.caption("USREC is the FRED representation of recession periods identified by the National Bureau of Economic Research (NBER).")
        st.line_chart(rec.set_index("DATE")["value"])
    except Exception:
        pass

    st.subheader("Master Dashboard — Four Macro Pillars")
    pillar_cols = st.columns(4)
    for col, (pillar, indicators) in zip(pillar_cols, US_MACRO_PILLARS.items()):
        with col:
            st.markdown(f"**{pillar}**")
            st.caption(indicators)

    category = st.selectbox("Indicator category", list(US_MACRO_INDICATORS.keys()), key="us_macro_category")
    indicator_name = st.selectbox("Indicator", list(US_MACRO_INDICATORS[category].keys()), key="us_macro_indicator")
    series_id = US_MACRO_INDICATORS[category][indicator_name]
    try:
        d = fred_series(series_id, "2000-01-01")
        latest = d.iloc[-1]
        previous = d.iloc[-2] if len(d) > 1 else latest
        change = float(latest["value"] - previous["value"]) if len(d) > 1 else np.nan
        direction = "↑ Rising" if change > 0 else "↓ Falling" if change < 0 else "→ Stable"
        a,b,c = st.columns(3)
        a.metric("Latest Value", f'{latest["value"]:,.3f}')
        b.metric("Change vs. Prior Observation", f"{change:,.3f}" if pd.notna(change) else "N/A")
        c.metric("Direction", direction)
        st.plotly_chart(px.line(d, x="DATE", y="value", title=f"U.S. {indicator_name} ({series_id})"), use_container_width=True)
        st.dataframe(d.tail(20), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"FRED data request failed for {series_id}: {e}")

    st.subheader("Professional Economic Analysis Workflow")
    workflow = [
        ("1","Business Cycle","Classify expansion, peak, contraction/recession evidence or trough using growth, labor, inflation and leading indicators."),
        ("2","Aggregate Demand / Supply","Determine whether major changes appear demand-driven, supply-driven or mixed."),
        ("3","Monetary Policy","Evaluate the Federal Funds Rate, balance sheet, Treasury curve and Federal Reserve communication."),
        ("4","Fiscal Policy","Evaluate federal spending, receipts, deficits, debt and Treasury financing."),
        ("5","Financial Markets","Analyze equities, rates, credit spreads, volatility, commodities and the dollar."),
        ("6","International Sector","Analyze trade, exchange rates, capital flows and global conditions."),
        ("7","Research Horizon","Build a documented 6–24 month scenario analysis for growth, inflation, unemployment, rates and asset markets.")
    ]
    st.dataframe(pd.DataFrame(workflow, columns=["Step","Analysis","Core Question"]), use_container_width=True, hide_index=True)
    st.caption("Outlooks should be expressed as scenarios with supporting evidence and uncertainty, not as guaranteed forecasts.")



def institutional_products_dashboard():
    st.header("Institutional Products & U.S. Securities Intelligence")
    st.caption(
        "Interactive research workspace for derivatives, structured products, fixed income and short-term funding instruments. "
        "Select a product concept, review its structure, then inspect the mapped U.S. security universe and market observations."
    )

    PRODUCT_GROUPS = {
        "Derivatives": [
            "Total Return Swaps (TRS)",
            "Credit Default Swaps (CDS)",
            "Interest Rate Swaps",
            "Currency Swaps",
            "Cross-Currency Swaps",
            "Equity Swaps",
            "Variance Swaps",
            "Volatility Swaps",
            "Inflation Swaps",
            "Commodity Swaps",
            "Swaptions",
            "SOFR Futures",
        ],
        "Structured Credit": [
            "Collateralized Loan Obligations (CLOs)",
            "Collateralized Debt Obligations (CDOs)",
            "Mortgage-Backed Securities (MBS)",
            "Residential Mortgage-Backed Securities (RMBS)",
            "Commercial Mortgage-Backed Securities (CMBS)",
            "Structured Notes",
        ],
        "Fixed Income & Funding": [
            "U.S. Treasuries",
            "Floating Rate Notes (FRNs)",
            "Convertible Bonds",
            "Commercial Paper",
            "Repurchase Agreements (Repos)",
        ],
    }

    flat_products = [(group, product) for group, items in PRODUCT_GROUPS.items() for product in items]
    left, right = st.columns([1, 2])
    with left:
        group = st.selectbox("Product group", list(PRODUCT_GROUPS.keys()), key="inst_product_group")
        product = st.selectbox("Product / concept", PRODUCT_GROUPS[group], key="inst_product")
    with right:
        st.subheader(product)
        st.write({
            "Product group": group,
            "Research focus": "Pricing, valuation, cash flows, market risk, credit/counterparty risk, liquidity, stress testing and regulatory intelligence.",
            "Security linkage": "Underlying, reference, collateral, issuer, benchmark or financing security as applicable."
        })

    profiles = {
        "Total Return Swaps (TRS)": ("Synthetic total-return exposure to a reference asset in exchange for financing or another agreed payment stream.", "Reference asset + financing leg + counterparty exposure"),
        "Credit Default Swaps (CDS)": ("Credit-risk transfer contract referencing an issuer or obligation.", "Reference entity/obligation + premium + protection payment"),
        "Interest Rate Swaps": ("Contract exchanging interest-rate cash flows, commonly fixed for floating.", "Notional + fixed leg + floating benchmark"),
        "Currency Swaps": ("Swap exchanging cash flows denominated in different currencies.", "Currency notionals + interest legs + FX exposure"),
        "Cross-Currency Swaps": ("Multi-currency funding or hedging contract exchanging principal and interest cash flows.", "Two currencies + funding curves + FX"),
        "Equity Swaps": ("Derivative providing economic exposure to equity returns without direct ownership.", "Equity/index return + financing leg"),
        "Variance Swaps": ("Contract transferring realized variance exposure against a fixed variance strike.", "Realized variance + variance strike"),
        "Volatility Swaps": ("Derivative whose payoff is linked directly to realized volatility.", "Realized volatility + volatility strike"),
        "Inflation Swaps": ("Contract exchanging inflation-linked cash flows for fixed or other reference payments.", "Inflation index + fixed leg"),
        "Commodity Swaps": ("Derivative exchanging commodity-linked payments for fixed or floating payments.", "Commodity reference price + swap terms"),
        "Swaptions": ("Options granting the right, but not obligation, to enter an interest-rate swap.", "Option premium + swap rate + volatility + rates curve"),
        "SOFR Futures": ("Exchange-traded futures contracts referencing the Secured Overnight Financing Rate.", "SOFR reference + contract convention + futures price"),
        "Collateralized Loan Obligations (CLOs)": ("Structured securities backed primarily by pools of leveraged loans with tranched cash flows.", "Loan collateral + waterfall + tranches + manager"),
        "Collateralized Debt Obligations (CDOs)": ("Structured credit vehicles issuing tranches against diversified debt exposures.", "Collateral pool + waterfall + tranche structure"),
        "Mortgage-Backed Securities (MBS)": ("Securities backed by pools of mortgage loans and their associated cash flows.", "Mortgage collateral + prepayments + interest + principal"),
        "Residential Mortgage-Backed Securities (RMBS)": ("MBS backed specifically by residential mortgage exposures.", "Residential mortgages + prepayment/default behavior"),
        "Commercial Mortgage-Backed Securities (CMBS)": ("Securities backed by commercial real-estate mortgage cash flows.", "Commercial property loans + property cash flows"),
        "Structured Notes": ("Debt securities whose payoff is linked to rates, equities, commodities, currencies or other reference variables.", "Issuer debt + embedded derivative"),
        "U.S. Treasuries": ("Debt obligations issued by the U.S. Treasury across bills, notes and bonds.", "Par value + coupon/discount + Treasury curve"),
        "Floating Rate Notes (FRNs)": ("Debt securities whose coupon resets periodically using a reference rate plus a spread.", "Reference rate + spread + reset schedule"),
        "Convertible Bonds": ("Corporate debt securities containing an option to convert into equity under specified terms.", "Bond cash flows + embedded equity option"),
        "Commercial Paper": ("Short-term unsecured corporate debt used primarily for working-capital financing.", "Issuer credit + maturity + discount/yield"),
        "Repurchase Agreements (Repos)": ("Short-term secured financing structured as a sale and later repurchase of securities.", "Collateral security + repo rate + haircut + maturity"),
    }
    definition, linkage = profiles[product]

    st.markdown("### Product profile")
    a,b,c = st.columns(3)
    a.metric("Product", product)
    b.metric("Structure", group)
    c.metric("Primary linkage", linkage)
    st.write(definition)

    tabs = st.tabs(["Security Universe", "Visualization", "Analytics", "Regulatory / Research"])
    with tabs[0]:
        st.subheader("U.S. base securities and reference instruments")
        # Reuse the concept security engine with product-specific concept text.
        concept_security_panel(product)

    with tabs[1]:
        st.subheader("Product market visualization")
        st.info("Select a mapped U.S. security in the Security Universe tab to populate the market history and risk visualization.")
        st.markdown("**Core visualization set:** price/value history • return • volatility • drawdown • volume/liquidity where available • comparative benchmark.")
        if product in {"U.S. Treasuries","Floating Rate Notes (FRNs)","Commercial Paper","Repurchase Agreements (Repos)"}:
            st.markdown("**Fixed-income/funding lens:** yield/price relationship • duration • spread • curve exposure • funding/liquidity context.")
        elif group == "Structured Credit":
            st.markdown("**Structured-credit lens:** collateral performance • tranche/waterfall exposure • spread • prepayment/default assumptions • scenario sensitivity.")
        else:
            st.markdown("**Derivatives lens:** underlying/reference asset • implied/realized risk • basis • counterparty exposure • scenario payoff.")

    with tabs[2]:
        st.subheader("Research analytics")
        metrics = {
            "TRS / Equity Swaps": ["Underlying return","Financing cost","Total return","Counterparty exposure","Basis risk"],
            "CDS": ["Credit spread","Default probability","Recovery assumption","CS01","Counterparty exposure"],
            "Swaps / Swaptions": ["Par swap rate","DV01","Duration","Convexity","Volatility / vega"],
            "SOFR Futures": ["Implied rate","Price change","Curve position","DV01","Basis"],
            "CLOs / CDOs": ["Collateral quality","Spread","Default sensitivity","Recovery","Tranche attachment/detachment"],
            "MBS / RMBS / CMBS": ["Yield","Duration","Convexity","Prepayment sensitivity","Credit spread"],
            "Structured Notes": ["Reference return","Embedded option value","Issuer credit","Payoff scenarios","Liquidity"],
            "Treasuries": ["Yield","Duration","DV01","Convexity","Curve spread"],
            "FRNs": ["Reference rate","Spread","Reset risk","Credit spread","Duration"],
            "Convertibles": ["Bond floor","Conversion value","Delta","Credit spread","Equity volatility"],
            "Commercial Paper": ["Discount/yield","Maturity","Issuer credit","Liquidity","Spread"],
            "Repos": ["Repo rate","Haircut","Collateral value","Funding cost","Counterparty exposure"],
        }
        key = next((k for k in metrics if k in product or product in k), "Treasuries")
        st.dataframe(pd.DataFrame({"Analytical metric": metrics[key]}), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Regulatory and institutional research")
        st.write({
            "Regulatory scope": "Map by instrument, transaction venue, counterparty, issuer and jurisdiction.",
            "Research sources": "SEC, CFTC, Federal Reserve, FINRA, OCC, FDIC and authorized commercial research/data providers where applicable.",
            "Point-in-time control": "Preserve publication/availability timestamps so historical analysis and ML features do not use information unavailable at the prediction time.",
        })
        st.caption("Commercial identifiers, transaction-level data and proprietary analytics must be resolved through licensed/reference-data sources rather than guessed.")

def ratings_research_intelligence():
    institutional_products_dashboard()
    st.header("Ratings, Research & Financial Intelligence")
    st.caption("Institutional research layer for credit ratings, equity research, market intelligence, bank research, ESG, risk providers, regulatory reports and financial instruments.")
    tabs = st.tabs(["Providers","Credit Ratings","Regulatory Intelligence","API Architecture","Products"])

    with tabs[0]:
        st.subheader("Institutional information-provider registry")
        providers = pd.DataFrame(provider_dataframe())
        categories = ["All"] + sorted(providers["category"].unique().tolist())
        cat = st.selectbox("Provider category", categories, key="provider_category")
        search = st.text_input("Search provider", key="provider_search")
        view = providers if cat == "All" else providers[providers["category"] == cat]
        if search.strip():
            view = view[view["name"].str.contains(search.strip(), case=False, na=False)]
        st.metric("Providers in registry", len(view))
        st.dataframe(view[["name","category","focus","priority","access"]], use_container_width=True, hide_index=True)
        st.download_button("Export provider registry", view.to_csv(index=False).encode(), "globalblocs_research_providers.csv", "text/csv")
        st.subheader("Core institutional research set")
        st.dataframe(pd.DataFrame(priority_provider_dataframe())[["name","category","focus","access"]], use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Credit-rating intelligence")
        st.write("Ratings should be treated as source observations rather than as GlobalBLOCS recommendations. Preserve the agency, rating action, rating date, security/issuer, outlook/watch status, methodology and source document.")
        rating_scale = pd.DataFrame([
            ["Investment grade","AAA / AA / A / BBB","Higher relative credit quality under the agency's methodology"],
            ["Speculative grade","BB / B / CCC / CC / C","Higher relative default/credit-risk assessment under the agency's methodology"],
            ["Default / distressed","D / RD or agency-specific equivalent","Use the exact agency definition; scales differ"],
        ], columns=["Grouping","Illustrative ratings","Research interpretation"])
        st.dataframe(rating_scale, use_container_width=True, hide_index=True)
        st.info("Agency scales and definitions are not perfectly interchangeable. GlobalBLOCS should retain the original agency rating and separately map it to a normalized analytical scale.")
        templates = pd.DataFrame(RATING_API_TEMPLATES)
        st.dataframe(templates[["provider","auth","endpoint","example_query"]], use_container_width=True, hide_index=True)
        st.warning("Commercial rating APIs are entitlement-controlled. Endpoint URLs and authentication flows shown here are configuration placeholders, not claims that the listed endpoints are publicly callable.")

        provider = st.selectbox("Provider for request template", templates["provider"].tolist(), key="rating_provider")
        endpoint = st.text_input("Licensed endpoint", "", key="rating_endpoint")
        ticker = st.text_input("Issuer / ticker", "MSFT", key="rating_issuer")
        if st.button("Generate credentialed request template", key="generate_rating_request"):
            if not endpoint.strip():
                st.warning("Enter the endpoint supplied by the licensed provider documentation.")
            else:
                request_template = credentialed_request_template(provider, endpoint.strip(), params={"issuer_or_ticker": ticker})
                st.code(
                    "import requests\\n\\n"
                    f"url = {request_template['endpoint']!r}\\n"
                    f"headers = {request_template['headers']!r}\\n"
                    f"params = {request_template['params']!r}\\n"
                    "response = requests.get(url, headers=headers, params=params, timeout=30)\\n"
                    "response.raise_for_status()\\n"
                    "data = response.json()",
                    language="python"
                )

    with tabs[2]:
        st.subheader("U.S. Financial Regulatory Intelligence")
        regs = pd.DataFrame(regulatory_dataframe())
        agency = st.selectbox("Regulatory agency", regs["agency"].tolist(), key="reg_agency")
        selected_reg = regs[regs["agency"] == agency].iloc[0]
        a,b,c = st.columns(3)
        a.metric("Agency", agency.split(" (")[0])
        b.metric("Primary reports", selected_reg["reports"])
        c.metric("Access", selected_reg["public_access"])
        st.dataframe(regs[["agency","reports","purpose","public_access"]], use_container_width=True, hide_index=True)
        st.subheader("Regulatory data-source configuration")
        source_name = st.selectbox("Source configuration", list(REGULATORY_SOURCE_CONFIG.keys()), key="reg_source")
        cfg = REGULATORY_SOURCE_CONFIG[source_name]
        st.json(cfg)
        st.info("Restricted datasets such as SAR filings are not treated as public ingestion targets. GlobalBLOCS should ingest only data for which the user has lawful authorization and the applicable provider terms permit automated use.")

        report = st.selectbox("Research report", [
            "SEC 10-K","SEC 10-Q","SEC 8-K","SEC Form 4","SEC 13F","FDIC Call Reports",
            "Federal Reserve FR Y-9C","Federal Reserve Z.1","CFTC COT","FINRA Short Interest",
            "CFPB Consumer Complaint Database","PCAOB Inspection Reports","OCC Quarterly Banking Profile",
            "FDIC Bank Failure Data"
        ], key="reg_report")
        report_use = {
            "SEC 10-K":"Annual company financial position, performance, cash flow and risk disclosures.",
            "SEC 10-Q":"Quarterly company financial and risk disclosures.",
            "SEC 8-K":"Material current events and corporate disclosures.",
            "SEC Form 4":"Insider transaction disclosures.",
            "SEC 13F":"Quarterly institutional investment-manager holdings disclosures.",
            "FDIC Call Reports":"Bank assets, liabilities, capital, earnings and loan information.",
            "Federal Reserve FR Y-9C":"Bank holding company consolidated financial information.",
            "Federal Reserve Z.1":"Financial Accounts of the United States and sectoral balance-sheet/flow relationships.",
            "CFTC COT":"Futures positioning by trader categories.",
            "FINRA Short Interest":"Reported short-interest information, subject to dataset definitions and timing.",
            "CFPB Consumer Complaint Database":"Consumer-finance complaint records and trend analysis.",
            "PCAOB Inspection Reports":"Public audit-inspection findings and audit-quality information.",
            "OCC Quarterly Banking Profile":"National-bank profitability, credit quality, capital and risk indicators.",
            "FDIC Bank Failure Data":"Historical bank-failure events and institution information.",
        }
        st.write(report_use[report])

    with tabs[3]:
        st.subheader("Production ingestion architecture")
        architecture = [
            ["1","Licensed / official source","Ratings, research, SEC, FDIC, Federal Reserve, CFTC, FINRA, CFPB and other permitted datasets"],
            ["2","Source connector","Credential handling, rate limits, retries, pagination and provider-specific schemas"],
            ["3","Raw / Bronze","Immutable source payload plus retrieval timestamp and source metadata"],
            ["4","Silver / normalized","Canonical issuer, security, rating, filing, report and observation structures"],
            ["5","PostgreSQL","Core financial data, metadata, provenance, compliance and analytics schemas"],
            ["6","Gold / intelligence","350 metrics, trends, risk, credit, valuation, macro and cross-source analytics"],
            ["7","ML / Deep Learning","Feature engineering, point-in-time joins, walk-forward validation and model monitoring"],
            ["8","Dashboard / API","Research UI, explain-result views, exports and downstream applications"],
        ]
        st.dataframe(pd.DataFrame(architecture, columns=["Layer","Component","Purpose"]), use_container_width=True, hide_index=True)
        st.subheader("Point-in-time and provenance requirements")
        st.dataframe(pd.DataFrame([
            ["provider","Original provider / agency","Required"],
            ["entity_id","Canonical issuer / company / institution identifier","Required"],
            ["security_id","Security identifier when applicable","Recommended"],
            ["rating","Original agency rating or research value","Required for ratings"],
            ["rating_date","Date of rating action / observation","Required"],
            ["publication_date","When the report became public/available","Required"],
            ["available_to_market_date","Earliest time the data could have been used","Required for ML"],
            ["retrieval_timestamp","When GlobalBLOCS retrieved it","Required"],
            ["source_document","Filing/report/document identifier","Required"],
            ["methodology_version","Provider methodology/model version when available","Recommended"],
            ["quality_grade","A/B/C/D/E reconciliation or quality class","Required"],
        ], columns=["Field","Definition","Requirement"]), use_container_width=True, hide_index=True)
        st.info("For research and ML, never join a later restatement or rating action into an earlier prediction window. Point-in-time availability is part of the feature definition.")

    with tabs[4]:
        st.subheader("50 Financial Products — Research Catalog")
        products = pd.DataFrame(product_dataframe())
        pcat = st.selectbox("Product category", ["All"] + sorted(products["category"].unique()), key="product_category")
        pv = products if pcat == "All" else products[products["category"] == pcat]
        st.metric("Products in catalog", len(pv))
        st.dataframe(pv, use_container_width=True, hide_index=True)
        st.download_button("Export product catalog", pv.to_csv(index=False).encode(), "globalblocs_50_financial_products.csv", "text/csv")
        selected_product = st.selectbox("Explain product", pv["product"].tolist(), key="selected_financial_product")
        product_row = pv[pv["product"] == selected_product].iloc[0]
        st.write({
            "Product": product_row["product"],
            "Category": product_row["category"],
            "Primary research uses": "Pricing, valuation, hedging, credit risk, liquidity, stress testing, scenario analysis and portfolio exposure.",
            "GlobalBLOCS treatment": "Instrument metadata → market/credit data → risk metrics → scenario analysis → portfolio and macro context."
        })
        st.subheader("Graduate-Level Product Profile")
        st.caption("Standardized research schema for the institutional-finance reference manual and future pricing/risk engines.")
        profile_template = {
            "Definition":"Instrument description, economic exposure and contractual structure.",
            "Purpose":"Financing, hedging, investment, liquidity or risk-transfer objective.",
            "How it works":"Trade initiation, valuation, collateral/margin and settlement mechanics.",
            "Parties involved":"Counterparties, issuers, investors, dealers, clearing parties and servicers as applicable.",
            "Cash flows":"Premiums, coupons, floating/reference payments, principal, collateral and settlement flows.",
            "Pricing methodology":"Discounted cash flow, no-arbitrage, option, curve, spread, model or market-comparable framework as appropriate.",
            "Risk characteristics":"Market, credit, counterparty, liquidity, basis, model, legal, operational and settlement risks.",
            "Return characteristics":"Coupon, carry, spread, capital gain/loss, optionality or leveraged exposure.",
            "Typical buyers and sellers":"Institutional investors, banks, dealers, funds, corporations, governments or eligible counterparties.",
            "Regulatory oversight":"Applicable securities, derivatives, banking, prudential, clearing and jurisdictional requirements.",
            "Real-world applications":"Hedging, financing, asset-liability management, portfolio construction and risk transfer.",
            "Python pricing / analytics":"Transparent educational pricing, sensitivity, scenario or risk calculation where applicable.",
            "Market-data identifiers":"Provider-specific identifiers only when licensed/available; never invent identifiers.",
            "Related regulatory filings":"Relevant public filings, reports, disclosures and transaction data where legally usable.",
            "Advantages":"Potential structural or economic benefits, described without investment recommendations.",
            "Disadvantages":"Costs, complexity, leverage, liquidity and structural limitations.",
            "Historical examples":"Documented links to episodes such as LTCM, 2008, Archegos or SVB when factually established."
        }
        st.dataframe(pd.DataFrame([profile_template]).T.rename(columns={0:"Research Profile Template"}), use_container_width=True)
        st.info("Identifiers and regulatory mappings are provider- and jurisdiction-specific and should be resolved from licensed/reference-data sources rather than guessed.")
\ndef global_economy():
    st.header("Global Economy & Economic BLOCs")
    st.caption("Select a world region or economic BLOC, then analyze macroeconomic conditions, markets, exchanges and financial metrics.")
    bloc = st.selectbox("Economic BLOC", ["None"] + list(ECONOMIC_BLOCS.keys()))
    region = st.selectbox("Region", list(REGIONS.keys()))
    indicator = st.selectbox("Map indicator", list(MAP_INDICATORS.keys()))
    year = st.slider("Map year", 2000, date.today().year, date.today().year-1)
    members = bloc_members(bloc, region)
    if members:
        st.info(f"Active geography: {bloc if bloc != 'None' else region} • {len(members)} selected economies")
    if st.button("Load global economic map", type="primary"):
        try:
            d = worldbank_all(MAP_INDICATORS[indicator], year)
            if members:
                d = d[d.iso3.isin(members)]
            fig = map_figure(d, f"{indicator} — {year}")
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.dataframe(d.sort_values("value", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Export map data", d.to_csv(index=False).encode(), "global_economy_map.csv", "text/csv")
        except Exception as e:
            st.error(f"Global map request failed: {e}")

    st.subheader("Applied economic concepts")
    cycle_country=st.selectbox("Economy to analyze",["USA","CHN","DEU","JPN","GBR","IND","CAN","BRA","AUS","KOR","MEX"],key="cycle_country")
    if st.button("Apply Business Cycle concept",type="secondary"):
        business_cycle_panel(cycle_country,cycle_country)

    st.subheader("Recession Intelligence Dashboard")
    st.caption("Country and Economic BLOC recession evidence is evaluated from macroeconomic and activity indicators. Results are transparent analytical screens, not official recession-dating decisions.")

    recession_country=st.selectbox("Country for recession analysis",["USA","CHN","DEU","JPN","GBR","IND","CAN","BRA","AUS","KOR","MEX","FRA","ITA","ESP","SGP","SAU","ARE"],key="recession_country")
    recession_scope = st.radio("Recession analysis scope",["Country","Economic BLOC"],horizontal=True,key="recession_scope")

    if recession_scope == "Country":
        if st.button("Analyze country recession evidence",type="secondary",key="analyze_country_recession"):
            recession_intelligence_panel(recession_country,recession_country)
    elif bloc == "None":
        st.info("Select an Economic BLOC above to enable BLOC-wide recession analysis.")
    else:
        st.write(f"Selected BLOC: **{bloc}** • {len(members)} member economies")
        if st.button(f"Analyze {bloc} recession evidence",type="secondary",key="analyze_bloc_recession"):
            bloc_recession_panel(bloc,members)

    with st.expander("Recession evidence framework", expanded=False):
        st.dataframe(pd.DataFrame([
            ["Real GDP growth","↓","High","Core output contraction/expansion signal"],
            ["GDP per-capita growth","↓","High","Per-person economic activity"],
            ["Household consumption","↓","Medium","Household demand"],
            ["Investment","↓","Medium","Business/capital formation"],
            ["Unemployment trend","↑","High","Labor-market weakening"],
            ["Manufacturing / industry","↓","Medium","Production-cycle evidence"],
            ["Services","↓","Medium","Service-sector activity"],
            ["Trade","↓","Medium","External-demand conditions"],
            ["Inflation","Context","Context","Separates demand weakness from price dynamics"],
            ["Leading indicators","↓","High","Future turning-point evidence when available"],
        ], columns=["Evidence","Recession direction","Weight","Interpretation"]), use_container_width=True, hide_index=True)
        st.info("GlobalBLOCS separates official recession determinations, GDP-rule screens, broad recession evidence, business-cycle phase, and leading recession risk. The current international screen primarily uses annual World Bank data; quarterly official series can be added as higher-frequency evidence.")
    st.subheader("Global market and exchange analysis")
    tickers = st.text_input("Market/exchange tickers", "AAPL,MSFT,NVDA,TSM,7203.T,005930.KS")
    domain = st.selectbox("Financial-intelligence domain", [x["name"] for x in FINANCIAL_DOMAINS])
    if st.button("Analyze selected markets", type="secondary"):
        d = bloc_market_summary(tickers, domain)
        st.dataframe(d, use_container_width=True, hide_index=True)
        if not d.empty:
            st.plotly_chart(px.bar(d, x="Ticker", y="Annual Volatility", title=f"{domain}: market volatility"), use_container_width=True)

def financial_domains():
    st.header("12 Financial Intelligence Domains")
    st.caption("Each domain is an end-user research module rather than a static description.")
    names = [x["name"] for x in FINANCIAL_DOMAINS]
    selected = st.selectbox("Domain", names)
    domain = next(x for x in FINANCIAL_DOMAINS if x["name"] == selected)
    a,b = st.columns(2)
    a.subheader(domain["name"]); a.write(domain["focus"])
    b.subheader("Core analytics"); b.write(", ".join(domain["metrics"]))
    tickers = st.text_input("Analyze securities/exchanges", "AAPL,MSFT,NVDA,TSM", key="domain_tickers")
    if st.button("Run domain analysis", type="primary"):
        d = bloc_market_summary(tickers, selected)
        if d.empty: st.warning("No market data returned for the selected instruments.")
        else:
            st.dataframe(d, use_container_width=True, hide_index=True)
            numeric = [x for x in ["1Y Return","Annual Volatility","Sharpe","Max Drawdown"] if x in d.columns]
            if numeric:
                metric = st.selectbox("Domain chart metric", numeric)
                st.plotly_chart(px.bar(d, x="Ticker", y=metric, title=f"{selected}: {metric}"), use_container_width=True)
    st.subheader("Global macro lens")
    macro_indicator = st.selectbox("Macro indicator", list(MAP_INDICATORS.keys()), key="domain_macro")
    macro_year = st.slider("Macro year", 2000, date.today().year, date.today().year-1, key="domain_year")
    if st.button("Load domain macro data"):
        try:
            d = worldbank_all(MAP_INDICATORS[macro_indicator], macro_year)
            st.plotly_chart(map_figure(d, f"{selected} — {macro_indicator} — {macro_year}"), use_container_width=True)
        except Exception as e:
            st.error(f"Macro data request failed: {e}")

def concept_security_panel(concept_name):
    """Market visualization and U.S.-listed security explorer for an economic concept."""
    rows = securities_for_concept(concept_name)
    securities = pd.DataFrame(rows)
    st.subheader("U.S. Base-Security Market Lens")
    st.caption(
        "The security list is the GlobalBLOCS U.S.-listed research universe mapped to the selected concept. "
        "It is a market-observation layer, not a claim that every security is causally determined by the concept. "
        "Direct Treasury/CUSIP security-master coverage requires an authorized reference-data source."
    )
    if securities.empty:
        st.info("No securities are currently mapped to this concept.")
        return

    groups = ["All"] + sorted(securities["security_group"].unique().tolist())
    group = st.selectbox("Security group", groups, key=f"concept_security_group_{concept_name}")
    view = securities if group == "All" else securities[securities["security_group"] == group]
    st.metric("Mapped U.S.-listed securities", len(view))

    # Scrolling/list view for research selection.
    st.dataframe(
        view[["ticker","security_group","concept"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    selected_ticker = st.selectbox(
        "Security to analyze",
        view["ticker"].tolist(),
        key=f"concept_security_ticker_{concept_name}",
    )
    period = st.selectbox(
        "Market history",
        ["3mo","6mo","1y","2y","5y"],
        index=2,
        key=f"concept_security_period_{concept_name}",
    )
    try:
        d = market_data(selected_ticker, period)
        if d.empty:
            st.warning(f"No market data returned for {selected_ticker}.")
            return
        close = d["Close"].dropna()
        ret = close.pct_change().dropna()
        latest = float(close.iloc[-1])
        total_return = float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else np.nan
        volatility = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else np.nan
        drawdown = float((close / close.cummax() - 1).min()) if len(close) else np.nan
        a,b,c,dcol = st.columns(4)
        a.metric("Latest Price", f"{latest:,.2f}")
        b.metric("Period Return", f"{total_return*100:.2f}%" if pd.notna(total_return) else "N/A")
        c.metric("Annualized Volatility", f"{volatility*100:.2f}%" if pd.notna(volatility) else "N/A")
        dcol.metric("Maximum Drawdown", f"{drawdown*100:.2f}%" if pd.notna(drawdown) else "N/A")

        st.plotly_chart(
            px.line(
                d.reset_index(),
                x=d.reset_index().columns[0],
                y="Close",
                title=f"{concept_name} → {selected_ticker}: price history",
            ),
            use_container_width=True,
        )

        summary = pd.DataFrame([{
            "Ticker": selected_ticker,
            "Concept": concept_name,
            "Security Group": group if group != "All" else "Mapped concept universe",
            "Latest Price": latest,
            "Period Return": total_return,
            "Annualized Volatility": volatility,
            "Maximum Drawdown": drawdown,
        }])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected-security history",
            d.reset_index().to_csv(index=False).encode(),
            f"{selected_ticker}_{concept_name.replace(' ','_')}.csv",
            "text/csv",
            key=f"concept_security_download_{concept_name}",
        )
    except Exception as e:
        st.error(f"Security analysis failed for {selected_ticker}: {e}")


def economics_concepts():
    st.header("100 Macroeconomic + 100 Microeconomic Concepts")
    st.caption(
        "Concept-by-concept research workspace: select an economic concept, view its formula and data source, "
        "visualize its historical series when available, and inspect the mapped U.S.-listed securities used as market observations."
    )
    kind = "Macroeconomics" if st.session_state.get("concept_mode","Macroeconomics") == "Macroeconomics" else "Microeconomics"
    concepts = MACRO_CONCEPTS if kind == "Macroeconomics" else MICRO_CONCEPTS
    categories = ["All"] + sorted({x["category"] for x in concepts})
    category = st.selectbox("Concept category", categories, key=f"econ_concept_category_{kind}")
    search = st.text_input("Search concepts", key=f"econ_concept_search_{kind}")
    d = pd.DataFrame(concepts)
    if category != "All":
        d = d[d.category == category]
    if search.strip():
        d = d[d.name.str.contains(search.strip(), case=False, na=False)]
    st.metric("Available concepts", len(d))
    st.dataframe(d[["id","name","category","formula"]], use_container_width=True, hide_index=True, height=320)
    st.download_button(
        "Export concept library",
        d.to_csv(index=False).encode(),
        f"{kind.lower()}_concepts.csv",
        "text/csv",
        key=f"concept_export_{kind}",
    )

    st.subheader("Concept Analysis")
    if d.empty:
        st.info("No concepts match the current filter.")
        return

    selected_name = st.selectbox(
        "Select concept for analysis",
        d["name"].tolist(),
        key=f"selected_economic_concept_{kind}",
    )
    selected = next(x for x in concepts if x["name"] == selected_name)
    a,b,c = st.columns(3)
    a.metric("Concept ID", selected.get("id","—"))
    b.metric("Category", selected.get("category","—"))
    c.metric("Mapped World Bank indicator", selected.get("indicator") or "Specialized / calculated")

    st.write({
        "Definition / Formula": selected["formula"],
        "World Bank indicator": selected.get("indicator") or "Concept requires a specialized calculation/data source.",
    })

    indicator_code = selected.get("indicator")
    if selected_name == "Business Cycle":
        country = st.selectbox(
            "ISO-3 economy",
            ["USA","CHN","DEU","JPN","GBR","IND","CAN","BRA","AUS","KOR","MEX"],
            key=f"concept_business_cycle_{kind}",
        )
        if st.button("Apply Business Cycle concept", type="primary", key=f"apply_business_cycle_{kind}"):
            business_cycle_panel(country, country)
            recession_intelligence_panel(country, country)
    elif indicator_code:
        country = st.text_input(
            "ISO-3 country code",
            "USA",
            max_chars=3,
            key=f"concept_country_{kind}",
        ).upper().strip()
        if st.button("Calculate historical concept series", key=f"concept_series_{kind}"):
            try:
                series = worldbank_country(indicator_code, country)
                st.plotly_chart(
                    px.line(series, x="year", y="value", markers=True, title=f"{selected_name} — {country}"),
                    use_container_width=True,
                )
                st.dataframe(series.tail(20), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Concept data request failed: {e}")

    st.subheader("Concept → U.S. Security Mapping")
    st.write("Mapped security groups:", ", ".join(concept_security_categories(selected_name)))
    concept_security_panel(selected_name)
)
# GlobalBLOCS credentialed API ingestion page.
st.sidebar.markdown("---")
st.sidebar.subheader("Data Connections")
_api_page = st.sidebar.selectbox("Workspace", ["Dashboard", "Credentialed API Ingestion"], key="globalblocs_workspace")
if _api_page == "Credentialed API Ingestion":
    api_ingestion_dashboard()
else:
    overview()
