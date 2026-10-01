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
from global_bloc_finance.economic_concepts import MACRO_CONCEPTS, MICRO_CONCEPTS
from global_bloc_finance.investment_metrics import METRICS, calculate_metrics, metric_catalog
from global_bloc_finance.recession_intelligence import assess_country, assess_bloc
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

def global_economy():
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

    st.subheader("Country recession analysis")
    recession_country=st.selectbox("Country",["USA","CHN","DEU","JPN","GBR","IND","CAN","BRA","AUS","KOR","MEX"],key="recession_country")
    if st.button("Analyze country recession evidence",type="secondary"):
        recession_intelligence_panel(recession_country,recession_country)

    if bloc != "None":
        st.subheader("Economic BLOC recession analysis")
        if st.button(f"Analyze {bloc} recession evidence",type="secondary"):
            bloc_recession_panel(bloc,members)

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

def economics_concepts():
    st.header("100 Macroeconomic + 100 Microeconomic Concepts")
    kind = "Macroeconomics" if st.session_state.get("concept_mode","Macroeconomics") == "Macroeconomics" else "Microeconomics"
    concepts = MACRO_CONCEPTS if kind == "Macroeconomics" else MICRO_CONCEPTS
    categories = ["All"] + sorted({x["category"] for x in concepts})
    category = st.selectbox("Category", categories)
    search = st.text_input("Search concepts")
    d = pd.DataFrame(concepts)
    if category != "All": d = d[d.category == category]
    if search.strip(): d = d[d.name.str.contains(search.strip(), case=False, na=False)]
    st.metric("Available concepts", len(d))
    st.dataframe(d[["id","name","category","formula"]], use_container_width=True, hide_index=True)
    st.download_button("Export concept library", d.to_csv(index=False).encode(), f"{kind.lower()}_concepts.csv", "text/csv")
    st.subheader("Apply a concept to a global economy")
    if not d.empty:
        selected_name = st.selectbox("Select concept", d["name"].tolist())
        selected = next(x for x in concepts if x["name"] == selected_name)
        st.write({"Definition / Formula": selected["formula"], "World Bank indicator": selected.get("indicator") or "Concept requires a specialized calculation/data source."})
        indicator_code = selected.get("indicator")
        if selected_name == "Business Cycle":
            country = st.selectbox("ISO-3 economy", ["USA","CHN","DEU","JPN","GBR","IND","CAN","BRA","AUS","KOR","MEX"], key=f"concept_business_cycle_{kind}")
            if st.button("Apply Business Cycle concept", type="primary", key=f"apply_business_cycle_{kind}"):
                business_cycle_panel(country, country)
                recession_intelligence_panel(country, country)
        elif indicator_code:
            country = st.text_input("ISO-3 country code", "USA").upper().strip()
            if st.button("Calculate historical series"):
                try:
                    series = worldbank_country(indicator_code, country)
                    st.plotly_chart(px.line(series, x="year", y="value", markers=True, title=f"{selected_name} — {country}"), use_container_width=True)
                    st.dataframe(series.tail(20), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Concept data request failed: {e}")


MARKET_TYPES = {
    "Stock Market (Equities)": {
        "description":"Markets where ownership interests in companies are issued and traded.",
        "examples":"S&P 500, Nasdaq Composite, individual listed equities",
        "analytics":"Price/return analysis, valuation, profitability, growth, liquidity, volatility and factor exposure."
    },
    "Bond Market (Fixed Income)": {
        "description":"Markets for debt securities issued by governments, corporations and other borrowers.",
        "examples":"Treasuries, corporate bonds, municipal bonds",
        "analytics":"Yield, duration, convexity, spread, credit risk, default risk and curve analysis."
    },
    "Forex Market (FX)": {
        "description":"Markets where currencies are exchanged against one another.",
        "examples":"EUR/USD, USD/JPY, GBP/USD",
        "analytics":"Returns, volatility, correlation, carry, relative strength and macro sensitivity."
    },
    "Derivatives Market": {
        "description":"Markets for contracts whose value is linked to an underlying asset, rate, index or commodity.",
        "examples":"Options, futures, swaps and forwards",
        "analytics":"Payoff analysis, Greeks, implied volatility, term structure and scenario analysis."
    },
    "Commodities Market": {
        "description":"Markets for physical commodities and contracts linked to them.",
        "examples":"Crude oil, natural gas, gold, wheat",
        "analytics":"Price trends, seasonality, volatility, inventory/macro relationships and cross-asset correlations."
    },
    "Cryptocurrency Market": {
        "description":"Markets for digital assets and tokenized networks.",
        "examples":"Bitcoin, Ethereum and other digital assets",
        "analytics":"Returns, volatility, drawdown, liquidity, correlation, market structure and on-chain data when available."
    }
}

MARKET_TERMS = [
    ("Stock","Ownership interest in a company"),("Share","One unit of stock"),("Exchange","Organized marketplace for trading"),
    ("Ticker Symbol","Short identifier for a security"),("Index","Group of securities used to represent a market or segment"),
    ("Market Cap","Market value of equity"),("Liquidity","Ease of transacting without materially moving price"),
    ("Volatility","Magnitude of price variation"),("Bull Market","Sustained rising-price environment"),
    ("Bear Market","Sustained falling-price environment"),("Bid Price","Price a buyer is willing to pay"),
    ("Ask Price","Price at which a seller is willing to transact"),("Spread","Difference between bid and ask"),
    ("Order","Instruction to transact"),("Market Order","Order intended for immediate execution"),
    ("Limit Order","Order constrained by a specified price"),("Stop-Loss","Conditional order intended to limit a loss"),
    ("Take Profit","Conditional order intended to lock in a specified gain"),("Volume","Quantity traded"),
    ("Slippage","Difference between expected and executed price"),("Trend","General direction of a price series"),
    ("Support","Price area where prior buying activity has appeared"),("Resistance","Price area where prior selling activity has appeared"),
    ("Breakout","Move beyond a previously defined resistance area"),("Breakdown","Move below a previously defined support area"),
    ("Trendline","Line used to summarize directional price structure"),("Channel","Range bounded by two trendlines"),
    ("Consolidation","Period of relatively limited directional movement"),("Pullback","Temporary move against a prevailing trend"),
    ("Reversal","Change from one prevailing directional pattern to another"),("Moving Average","Average price calculated over a rolling window"),
    ("Exponential Moving Average","Moving average assigning greater weight to recent observations"),
    ("RSI","Momentum oscillator based on recent gains and losses"),("MACD","Momentum/trend indicator based on moving-average differences"),
    ("Bollinger Bands","Volatility bands around a moving average"),("Momentum","Rate of change in a price or return series"),
    ("Revenue","Company sales or operating income"),("Earnings","Company profit after expenses"),
    ("EPS","Earnings attributable per share"),("P/E","Price relative to earnings"),("Dividend","Distribution of company profits to shareholders"),
    ("Balance Sheet","Statement of assets, liabilities and equity"),("Cash Flow","Movement of cash through operating, investing and financing activities"),
    ("Growth Stock","Security associated with relatively high expected growth"),("Value Stock","Security characterized by valuation measures relative to fundamentals"),
    ("Diversification","Spreading exposure across assets or risk sources"),("Portfolio","Collection of investments"),
    ("Risk Management","Process of identifying, measuring and controlling risk"),("Leverage","Use of borrowed capital or other amplified exposure"),
    ("Margin","Collateralized borrowing or account requirement for trading"),("Hedging","Using an offsetting exposure to reduce a risk"),
    ("Drawdown","Decline from a prior peak"),("Risk-Reward Ratio","Comparison of potential gain and potential loss"),
    ("Market Sentiment","Aggregate market attitudes reflected in observable behavior"),("FOMO","Fear of missing out"),
    ("Panic Selling","Rapid selling associated with stressed market conditions"),("Institutional Buying","Observable purchases associated with large institutions"),
    ("Liquidity Zones","Price areas associated with elevated historical trading activity"),("Order Flow","Sequence and imbalance of buying and selling orders"),
    ("Smart Money","Informal term for sophisticated or institutional market participants")
]

def _trend_badge(direction):
    if direction in ("Trending Up","Improving"):
        return "🟢"
    if direction in ("Trending Down","Deteriorating"):
        return "🔴"
    if direction in ("Sideways","Stable"):
        return "🟡"
    return "⚪"

def investment_metrics_page():
    st.header("50 Investment Metrics")
    st.caption("Every metric is calculated from the selected security/portfolio where the required data are available. Direction describes movement of the metric; it is not a recommendation.")
    tickers = st.text_input("Security ticker(s), comma-separated", "AAPL", key="investment_metric_tickers").upper()
    benchmark = st.text_input("Benchmark ticker", "SPY", key="investment_metric_benchmark").upper().strip()
    period = st.selectbox("Calculation history", ["1y","2y","5y","10y","max"], index=1, key="investment_metric_period")
    category = st.selectbox("Metric category", ["All"] + sorted({x[2] for x in METRICS}), key="investment_metric_category")
    selected_names = [x[1] for x in METRICS if category == "All" or x[2] == category]
    selected_metric = st.selectbox("Explain a metric", selected_names, key="investment_metric_selected")
    if st.button("Calculate 50 investment metrics", type="primary"):
        symbols = [x.strip() for x in tickers.split(",") if x.strip()]
        if not symbols:
            st.warning("Enter at least one ticker.")
            return
        try:
            primary = symbols[0]
            primary_df = market_data(primary, period)
            bench_df = market_data(benchmark, period) if benchmark else None
            portfolio = pd.concat(
                [market_data(s, period).Close.rename(s) for s in symbols], axis=1
            ).dropna(how="all")
            result = calculate_metrics(primary_df, info(primary), bench_df, portfolio_prices=portfolio)
            if result.empty:
                st.warning("No usable market data were returned.")
                return
            result["Indicator"] = result["Direction"].map(_trend_badge) + " " + result["Direction"]
            shown = result[result["Category"].eq(category)] if category != "All" else result
            st.dataframe(
                shown[["Indicator","ID","Metric","Category","Value","Trend","Performance Note"]],
                use_container_width=True, hide_index=True
            )
            row = result[result.Metric.eq(selected_metric)].iloc[0]
            a,b,c,d = st.columns(4)
            a.metric("Metric", row["Metric"])
            value = row["Value"]
            b.metric("Current Value", "N/A" if pd.isna(value) else f"{value:,.4f}")
            c.metric("Direction", f'{_trend_badge(row["Direction"])} {row["Direction"]}')
            d.metric("Trend", row["Trend"])
            st.info(row["Definition"])
            st.write(row["Performance Note"])
            st.download_button("Download 50-metric results", result.to_csv(index=False).encode(), f"{primary}_50_investment_metrics.csv", "text/csv")
        except Exception as e:
            st.error(f"Metric calculation failed: {e}")

def financial_markets_page():
    st.header("Financial Markets")
    st.write("Financial markets connect buyers and sellers of financial assets. Global BLOC uses this section to organize market structure, participants, instruments, trends and analytical methods.")
    tabs = st.tabs(["Overview","Market Types","Participants","Structure","Trends & Analysis","Terms"])
    with tabs[0]:
        st.subheader("What are financial markets?")
        st.write("Financial markets are systems in which financial assets are issued, bought and sold. Their core economic functions include capital formation, liquidity and price discovery.")
        st.dataframe(pd.DataFrame([
            ["Raise capital","Companies and governments can obtain financing."],
            ["Create liquidity","Participants can buy or sell financial assets."],
            ["Price discovery","Transactions and information contribute to observable market prices."]
        ], columns=["Function","Description"]), use_container_width=True, hide_index=True)
    with tabs[1]:
        for name, item in MARKET_TYPES.items():
            with st.expander(name, expanded=False):
                st.write(item["description"])
                st.write("Examples:", item["examples"])
                st.write("Global BLOC analytics:", item["analytics"])
    with tabs[2]:
        st.dataframe(pd.DataFrame([
            ["Retail investors","Individuals investing or trading for their own accounts."],
            ["Institutional investors","Organizations such as pension funds, mutual funds, insurers and asset managers."],
            ["Market makers","Participants that facilitate liquidity by quoting buy and sell prices."],
            ["Brokers","Intermediaries that facilitate execution and market access."],
            ["Regulators","Public authorities that establish and enforce market rules."]
        ], columns=["Participant","Role"]), use_container_width=True, hide_index=True)
    with tabs[3]:
        a,b = st.columns(2)
        a.subheader("Primary Market")
        a.write("New securities are issued to raise capital, such as an initial public offering or a new bond issue.")
        b.subheader("Secondary Market")
        b.write("Previously issued securities are traded among market participants.")
    with tabs[4]:
        st.subheader("Trend and analytical framework")
        st.write("Uptrend: higher highs and higher lows. Downtrend: lower highs and lower lows. Sideways: relatively range-bound price behavior.")
        st.write("Fundamental analysis examines financial and economic conditions. Technical analysis examines price, volume and indicators. Sentiment analysis examines observable market behavior and positioning.")
        st.info("A trend indicator describes observed market structure. It does not establish that a future breakout, reversal or return is guaranteed.")
    with tabs[5]:
        term_search = st.text_input("Search market terms", key="market_term_search")
        td = pd.DataFrame(MARKET_TERMS, columns=["Term","Definition"])
        if term_search.strip():
            td = td[td.Term.str.contains(term_search.strip(), case=False, na=False) | td.Definition.str.contains(term_search.strip(), case=False, na=False)]
        st.dataframe(td, use_container_width=True, hide_index=True)

def market_type_page(market_name):
    item = MARKET_TYPES[market_name]
    st.header(market_name)
    st.write(item["description"])
    st.caption(item["examples"])
    ticker_defaults = {
        "Stock Market (Equities)":"AAPL,MSFT,NVDA",
        "Bond Market (Fixed Income)":"^TNX",
        "Forex Market (FX)":"EURUSD=X,JPY=X,GBPUSD=X",
        "Derivatives Market":"^VIX",
        "Commodities Market":"CL=F,GC=F,ZW=F",
        "Cryptocurrency Market":"BTC-USD,ETH-USD",
    }
    symbols = st.text_input("Market instruments", ticker_defaults[market_name], key=f"market_{market_name}")
    if st.button("Analyze market", type="primary", key=f"analyze_{market_name}"):
        rows=[]
        for s in [x.strip() for x in symbols.split(",") if x.strip()]:
            try:
                d=market_data(s,"2y")
                close=d.Close.dropna()
                r=close.pct_change().dropna()
                ann=r.std()*np.sqrt(252)
                total=close.iloc[-1]/close.iloc[0]-1
                trend="Uptrend" if close.tail(50).iloc[-1] > close.tail(50).iloc[0] else "Downtrend"
                rows.append({"Ticker":s,"Last":close.iloc[-1],"2Y Return":total,"Annualized Volatility":ann,"Trend":trend})
            except Exception as e:
                rows.append({"Ticker":s,"Error":str(e)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if rows:
            valid=[x for x in rows if "2Y Return" in x]
            if valid:
                st.plotly_chart(px.bar(pd.DataFrame(valid),x="Ticker",y="2Y Return",title=f"{market_name}: observed return"),use_container_width=True)
    st.subheader("Analytical capabilities")
    st.write(item["analytics"])
    st.write("The market page can feed the same metric, risk, trend, benchmark, macro and ML engines used elsewhere in Global BLOC.")

def metric_visualizations():
    st.header("350 Metrics — Visualization Workspace"); metric={"id":1,"name":"Simple Return","category":"return"}; opts=visualization_options(metric); chosen=st.selectbox("Visualization",opts,format_func=lambda x: ("* " if x["recommended"] else "")+x["name"]); st.caption(chosen["reason"]); st.write("Recommendations identify useful starting points; they do not restrict selection.")

pages={
    "Overview":overview,
    "Financial Markets":financial_markets_page,
    "Stock Market":lambda: market_type_page("Stock Market (Equities)"),
    "Bond Market":lambda: market_type_page("Bond Market (Fixed Income)"),
    "Forex Market":lambda: market_type_page("Forex Market (FX)"),
    "Derivatives Market":lambda: market_type_page("Derivatives Market"),
    "Commodities Market":lambda: market_type_page("Commodities Market"),
    "Cryptocurrency Market":lambda: market_type_page("Cryptocurrency Market"),
    "50 Investment Metrics":investment_metrics_page,
    "Global Economy":global_economy,
    "12 Intelligence Domains":financial_domains,
    "Economics":economics,
    "Macroeconomic Concepts":lambda: (st.session_state.update(concept_mode="Macroeconomics"), economics_concepts())[1],
    "Microeconomic Concepts":lambda: (st.session_state.update(concept_mode="Microeconomics"), economics_concepts())[1],
    "Stock Exchanges":exchanges,
    "Banking & Regulation":regulation,
    "Import / Export":trade,
    "Industry Sectors":sectors,
    "ML / Deep Learning":ml,
    "Data Explorer":explorer,
    "Installation":installation,
    "350 Metrics":metric_visualizations
}
selection=st.sidebar.radio("Global BLOC modules",list(pages)); pages[selection]()
