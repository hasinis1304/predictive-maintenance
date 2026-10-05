import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import hashlib
from datetime import datetime
from src.features import engineer

st.set_page_config(page_title="PredictX - Industrial Maintenance", page_icon="🏭", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
    .stApp { background: #f2f2f0; font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background: #2d2d2d; }
    [data-testid="stSidebar"] { background: #2d2d2d; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
    h1 { color: #2d2d2d !important; font-weight: 800 !important; }
    h2, h3 { color: #3a3a3a !important; font-weight: 700 !important; }
    .login-container { max-width:420px;margin:60px auto;background:#fff;border-radius:20px;padding:40px;box-shadow:0 10px 40px rgba(0,0,0,0.08);border-top:5px solid #ffc107; }
    .login-logo { text-align:center;font-size:3.5rem;margin-bottom:5px; }
    .login-title { text-align:center;font-size:1.6rem;font-weight:800;color:#2d2d2d;margin:0 0 4px 0; }
    .login-subtitle { text-align:center;font-size:0.85rem;color:#888;margin-bottom:25px; }
    .hero-banner { background:linear-gradient(135deg,#2d2d2d 0%,#3a3a3a 100%);border-left:6px solid #ffc107;border-radius:0 14px 14px 0;padding:22px 28px;margin-bottom:20px; }
    .hero-banner h1 { color:#ffc107 !important;margin:0 !important;font-size:1.7rem !important; }
    .hero-banner p { color:#ccc !important;margin:6px 0 0 0 !important;font-size:0.9rem; }
    .metric-card { background:#fff;border:1px solid #e0e0e0;border-top:4px solid #ffc107;border-radius:14px;padding:18px 20px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.05);transition:transform 0.2s,box-shadow 0.2s;margin-bottom:12px; }
    .metric-card:hover { transform:translateY(-3px);box-shadow:0 6px 20px rgba(0,0,0,0.1); }
    .metric-value { font-size:2rem;font-weight:800;color:#2d2d2d;margin:4px 0;font-family:'JetBrains Mono',monospace; }
    .metric-label { font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1.5px;font-weight:600; }
    .metric-value.green { color:#2e7d32; } .metric-value.red { color:#c62828; } .metric-value.yellow { color:#f9a825; } .metric-value.blue { color:#1565c0; }
    .status-healthy { background:#e8f5e9;border:2px solid #66bb6a;border-radius:14px;padding:18px;text-align:center; }
    .status-healthy h3 { color:#2e7d32 !important; }
    .status-danger { background:#fff3e0;border:2px solid #ffa726;border-radius:14px;padding:18px;text-align:center;animation:warningPulse 2s infinite; }
    .status-danger h3 { color:#e65100 !important; }
    @keyframes warningPulse { 0%{box-shadow:0 0 0 0 rgba(255,167,38,0.4);}70%{box-shadow:0 0 0 12px rgba(255,167,38,0);}100%{box-shadow:0 0 0 0 rgba(255,167,38,0);} }
    .status-critical { background:#ffebee;border:2px solid #ef5350;border-radius:14px;padding:18px;text-align:center;animation:dangerPulse 1.5s infinite; }
    .status-critical h3 { color:#c62828 !important; }
    @keyframes dangerPulse { 0%{box-shadow:0 0 0 0 rgba(239,83,80,0.5);}70%{box-shadow:0 0 0 15px rgba(239,83,80,0);}100%{box-shadow:0 0 0 0 rgba(239,83,80,0);} }
    .section-card { background:#fff;border:1px solid #e0e0e0;border-radius:14px;padding:22px;box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-bottom:14px; }
    .hazard-stripe { background:repeating-linear-gradient(-45deg,#ffc107,#ffc107 10px,#2d2d2d 10px,#2d2d2d 20px);height:5px;border-radius:3px;margin:12px 0; }
    .stTabs [data-baseweb="tab-list"] { gap:4px;background:#2d2d2d;border-radius:10px;padding:4px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px;color:#999;font-weight:600;font-size:0.9rem; }
    .stTabs [aria-selected="true"] { background:#ffc107 !important;color:#2d2d2d !important; }
    .stButton > button { background:#ffc107 !important;color:#2d2d2d !important;border:none !important;border-radius:10px !important;padding:10px 20px !important;font-weight:700 !important;font-size:0.95rem !important;transition:all 0.2s !important;text-transform:uppercase;letter-spacing:0.5px; }
    .stButton > button:hover { background:#ffca28 !important;transform:scale(1.02);box-shadow:0 4px 15px rgba(255,193,7,0.4) !important; }
    .stSlider > div > div > div { background:#ffc107 !important; }
    [data-testid="stMetric"] { background:#fff;border:1px solid #e0e0e0;border-top:3px solid #ffc107;border-radius:10px;padding:12px; }
    [data-testid="stMetricValue"] { color:#2d2d2d !important;font-weight:700 !important; }
    div[data-testid="stFileUploader"] { background:#fff;border:2px dashed #ffc107;border-radius:14px;padding:20px; }
    .history-row { background:#fff;border:1px solid #eee;border-radius:10px;padding:12px 16px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;transition:background 0.2s; }
    .history-row:hover { background:#fffde7; }
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "prediction_history" not in st.session_state: st.session_state.prediction_history = []
if "batch_history" not in st.session_state: st.session_state.batch_history = []
if "page" not in st.session_state: st.session_state.page = "Dashboard"
if "total_predictions" not in st.session_state: st.session_state.total_predictions = 0
if "total_alerts" not in st.session_state: st.session_state.total_alerts = 0

USERS = {"admin":hashlib.sha256("admin123".encode()).hexdigest(),"hasini":hashlib.sha256("hasini123".encode()).hexdigest(),"lakshanaa":hashlib.sha256("lakshanaa123".encode()).hexdigest(),"operator":hashlib.sha256("operator123".encode()).hexdigest(),"demo":hashlib.sha256("demo".encode()).hexdigest()}
USER_ROLES = {"admin":"Administrator","hasini":"ML Engineer","lakshanaa":"ML Engineer","operator":"Machine Operator","demo":"Guest"}

@st.cache_resource
def load_model():
    artifact = joblib.load("models/model.joblib")
    return artifact["model"], artifact["threshold"]

model, threshold = load_model()

def make_gauge(value):
    if value < 0.3: bar_color = "#4caf50"
    elif value < threshold: bar_color = "#ffc107"
    else: bar_color = "#ef5350"
    status = "NORMAL" if value < 0.3 else ("CAUTION" if value < threshold else "CRITICAL")
    fig = go.Figure(go.Indicator(mode="gauge+number",value=value*100,number={"suffix":"%","font":{"size":48,"color":"#2d2d2d","family":"Inter"}},gauge={"axis":{"range":[0,100],"tickwidth":2,"tickcolor":"#999","tickfont":{"color":"#666","size":11}},"bar":{"color":bar_color,"thickness":0.7},"bgcolor":"#f0f0f0","borderwidth":2,"bordercolor":"#ddd","steps":[{"range":[0,30],"color":"#e8f5e9"},{"range":[30,threshold*100],"color":"#fff8e1"},{"range":[threshold*100,100],"color":"#ffebee"}],"threshold":{"line":{"color":"#c62828","width":3},"thickness":0.85,"value":threshold*100}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260,margin=dict(l=25,r=25,t=25,b=10),annotations=[dict(text=status,x=0.5,y=-0.05,showarrow=False,font=dict(size=14,color=bar_color,family="Inter"),xref="paper",yref="paper")])
    return fig

def make_radar(fd):
    cats=list(fd.keys());vals=list(fd.values())
    mins={"type":0,"air_temp":295,"process_temp":305,"rpm":1100,"torque":3,"tool_wear":0,"temp_diff":7,"power_w":1000,"strain":0,"torque_per_rpm":0,"temp_rpm_ratio":0.1}
    maxs={"type":2,"air_temp":305,"process_temp":315,"rpm":2900,"torque":80,"tool_wear":260,"temp_diff":13,"power_w":11000,"strain":17000,"torque_per_rpm":0.07,"temp_rpm_ratio":0.27}
    n=[(v-mins.get(c,0))/(maxs.get(c,1)-mins.get(c,0)+1e-9) for c,v in zip(cats,vals)]
    fig=go.Figure();fig.add_trace(go.Scatterpolar(r=n+[n[0]],theta=cats+[cats[0]],fill="toself",fillcolor="rgba(255,193,7,0.15)",line=dict(color="#ffc107",width=2.5)))
    fig.update_layout(polar=dict(bgcolor="#fff",radialaxis=dict(visible=True,range=[0,1],gridcolor="#eee",tickfont=dict(color="#999",size=9)),angularaxis=dict(gridcolor="#eee",tickfont=dict(color="#555",size=10))),paper_bgcolor="rgba(0,0,0,0)",height=300,margin=dict(l=55,r=55,t=25,b=25),showlegend=False)
    return fig

def make_feature_bars(fd):
    df=pd.DataFrame({"Feature":list(fd.keys()),"Value":list(fd.values())})
    mins={"type":0,"air_temp":295,"process_temp":305,"rpm":1100,"torque":3,"tool_wear":0,"temp_diff":7,"power_w":1000,"strain":0,"torque_per_rpm":0,"temp_rpm_ratio":0.1}
    maxs={"type":2,"air_temp":305,"process_temp":315,"rpm":2900,"torque":80,"tool_wear":260,"temp_diff":13,"power_w":11000,"strain":17000,"torque_per_rpm":0.07,"temp_rpm_ratio":0.27}
    df["Normalized"]=df.apply(lambda r:(r["Value"]-mins.get(r["Feature"],0))/(maxs.get(r["Feature"],1)-mins.get(r["Feature"],0)+1e-9),axis=1)
    df["Color"]=df["Normalized"].apply(lambda x:"#4caf50" if x<0.5 else("#ffc107" if x<0.8 else "#ef5350"))
    fig=go.Figure(go.Bar(x=df["Normalized"],y=df["Feature"],orientation="h",marker=dict(color=df["Color"],line=dict(width=0)),text=df["Value"].apply(lambda x:str(round(x,1))),textposition="outside",textfont=dict(color="#555",size=11)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(range=[0,1.15],showgrid=True,gridcolor="#eee",zeroline=False,title="Normalized Value",tickfont=dict(color="#888")),yaxis=dict(tickfont=dict(color="#555",size=11),autorange="reversed"),height=320,margin=dict(l=110,r=40,t=10,b=40))
    return fig

def add_to_history(inputs,prob,status):
    st.session_state.prediction_history.insert(0,{"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"user":st.session_state.username,"type":inputs["Type"],"air_temp":inputs["Air temperature [K]"],"process_temp":inputs["Process temperature [K]"],"rpm":inputs["Rotational speed [rpm]"],"torque":inputs["Torque [Nm]"],"tool_wear":inputs["Tool wear [min]"],"probability":prob,"status":status})
    st.session_state.total_predictions += 1
    if status != "Healthy": st.session_state.total_alerts += 1

def login_page():
    st.markdown("");st.markdown("")
    _,col2,_=st.columns([1,1.2,1])
    with col2:
        st.markdown('<div class="login-container"><div class="login-logo">🏭</div><div class="login-title">PredictX</div><div class="login-subtitle">Industrial Predictive Maintenance System</div></div>',unsafe_allow_html=True)
        with st.form("login_form"):
            username=st.text_input("Username",placeholder="Enter username")
            password=st.text_input("Password",type="password",placeholder="Enter password")
            st.checkbox("Remember me")
            submitted=st.form_submit_button("SIGN IN",use_container_width=True)
            if submitted:
                if username in USERS:
                    if USERS[username]==hashlib.sha256(password.encode()).hexdigest():
                        st.session_state.logged_in=True;st.session_state.username=username;st.session_state.page="Dashboard";st.rerun()
                    else: st.error("Incorrect password")
                else: st.error("User not found")
        st.markdown("")
        with st.expander("Demo Credentials"):
            st.code("Username: demo\nPassword: demo",language="text")
            st.code("Username: admin\nPassword: admin123",language="text")
            st.code("Username: hasini\nPassword: hasini123",language="text")

def render_sidebar():
    with st.sidebar:
        st.markdown('<div style="text-align:center;padding:15px 0;"><span style="font-size:2rem;">🏭</span><h2 style="margin:5px 0 0 0;color:#ffc107 !important;">PredictX</h2><p style="font-size:0.8rem;color:#888 !important;margin:0;">Maintenance Intelligence</p></div>',unsafe_allow_html=True)
        st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
        role=USER_ROLES.get(st.session_state.username,"User")
        st.markdown(f'<div style="background:#3a3a3a;border-radius:10px;padding:12px;margin:10px 0;"><p style="margin:0;font-size:0.85rem;">&#128100; <b>{st.session_state.username}</b></p><p style="margin:2px 0 0 0;font-size:0.75rem;color:#ffc107 !important;">{role}</p></div>',unsafe_allow_html=True)
        st.markdown("")
        for page,icon in zip(["Dashboard","Predict","Batch Analysis","Live Monitor","History","Analytics","Settings"],["📊","🔍","📋","📈","📜","🧠","⚙️"]):
            if st.button(f"{icon}  {page}",use_container_width=True,key=f"nav_{page}"):
                st.session_state.page=page;st.rerun()
        st.markdown("");st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True);st.markdown("")
        st.markdown(f'<div style="background:#3a3a3a;border-radius:10px;padding:12px;"><p style="font-size:0.75rem;color:#888 !important;margin:0;">SESSION STATS</p><p style="margin:4px 0;font-size:0.85rem;">Predictions: <b style="color:#ffc107;">{st.session_state.total_predictions}</b></p><p style="margin:4px 0;font-size:0.85rem;">Alerts: <b style="color:#ef5350;">{st.session_state.total_alerts}</b></p></div>',unsafe_allow_html=True)
        st.markdown("")
        if st.button("Logout",use_container_width=True): st.session_state.logged_in=False;st.session_state.username="";st.rerun()

def page_dashboard():
    st.markdown('<div class="hero-banner"><h1>🏭 PredictX Dashboard</h1><p>Real-time overview of your predictive maintenance system</p></div>',unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.markdown('<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value green">99.3%</div></div>',unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value blue">96.6%</div></div>',unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value yellow">82.4%</div></div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Predictions Today</div><div class="metric-value">{st.session_state.total_predictions}</div></div>',unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="metric-label">Alerts Raised</div><div class="metric-value red">{st.session_state.total_alerts}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
    col_left,col_right=st.columns([3,2],gap="large")
    with col_left:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Dataset Overview")
        data=pd.read_csv("data/ai4i2020.csv");features=engineer(data);probs=model.predict_proba(features)[:,1]
        fig=go.Figure();fig.add_trace(go.Histogram(x=probs,nbinsx=60,marker_color="#ffc107",opacity=0.85));fig.add_vline(x=threshold,line_dash="dash",line_color="#c62828",annotation_text="Threshold")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",xaxis=dict(title="Failure Probability",gridcolor="#eee"),yaxis=dict(title="Machine Count",gridcolor="#eee"),height=300,margin=dict(l=50,r=20,t=30,b=50))
        st.plotly_chart(fig,use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Recent Predictions")
        if st.session_state.prediction_history:
            for rec in st.session_state.prediction_history[:5]:
                ic="&#128994;" if rec["status"]=="Healthy" else "&#128308;"
                st.markdown(f'<div class="history-row"><span>{ic} <b>Machine Type {rec["type"]}</b></span><span style="font-family:\'JetBrains Mono\',monospace;color:#555;">{rec["probability"]:.1%}</span><span style="color:#999;font-size:0.8rem;">{rec["timestamp"]}</span></div>',unsafe_allow_html=True)
        else: st.info("No predictions yet. Go to the Predict page to analyze a machine.")
        st.markdown('</div>',unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("System Status")
        ar=int((probs>=threshold).sum());h=len(probs)-ar
        fig=go.Figure(data=[go.Pie(labels=["Healthy","At Risk"],values=[h,ar],hole=0.65,marker=dict(colors=["#4caf50","#ef5350"],line=dict(color="#fff",width=2)),textinfo="label+percent",textfont=dict(size=13))])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=280,margin=dict(l=20,r=20,t=20,b=20),showlegend=False,annotations=[dict(text=f"<b>{ar}</b><br>At Risk",x=0.5,y=0.5,font_size=16,showarrow=False)])
        st.plotly_chart(fig,use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Quick Actions")
        if st.button("New Prediction",use_container_width=True,key="qa_p"): st.session_state.page="Predict";st.rerun()
        if st.button("Batch Upload",use_container_width=True,key="qa_b"): st.session_state.page="Batch Analysis";st.rerun()
        if st.button("Start Monitor",use_container_width=True,key="qa_m"): st.session_state.page="Live Monitor";st.rerun()
        if st.button("View Analytics",use_container_width=True,key="qa_a"): st.session_state.page="Analytics";st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

def page_predict():
    st.markdown('<div class="hero-banner"><h1>🔍 Machine Failure Prediction</h1><p>Enter sensor readings to predict whether a machine needs maintenance</p></div>',unsafe_allow_html=True)
    col_input,col_result=st.columns([2,3],gap="large")
    with col_input:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Sensor Readings");st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
        machine_type=st.selectbox("Machine Type",["L - Low Quality","M - Medium Quality","H - High Quality"])
        st.markdown("**Temperature**");c1,c2=st.columns(2);air_temp=c1.slider("Air Temp (K)",295.0,305.0,300.0,0.1);process_temp=c2.slider("Process Temp (K)",305.0,315.0,310.0,0.1)
        st.markdown("**Mechanical**");c3,c4=st.columns(2);rpm=c3.slider("Speed (RPM)",1100,2900,1500);torque=c4.slider("Torque (Nm)",3.0,80.0,40.0,0.1)
        st.markdown("**Wear**");tool_wear=st.slider("Tool Wear (min)",0,260,100)
        predict_btn=st.button("ANALYZE MACHINE",type="primary",use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Quick Presets")
        pr1,pr2,pr3=st.columns(3);ph=pr1.button("Healthy",use_container_width=True,key="pr_h");pw=pr2.button("Warning",use_container_width=True,key="pr_w");pc=pr3.button("Critical",use_container_width=True,key="pr_c")
        st.markdown('</div>',unsafe_allow_html=True)
    tc=machine_type.split(" - ")[0].strip()
    if ph: air_temp,process_temp,rpm,torque,tool_wear=300.0,310.0,1500,40.0,50;predict_btn=True
    elif pw: air_temp,process_temp,rpm,torque,tool_wear=302.0,311.5,1350,55.0,180;predict_btn=True
    elif pc: air_temp,process_temp,rpm,torque,tool_wear=303.5,312.0,1200,70.0,240;tc="L";predict_btn=True
    with col_result:
        if predict_btn:
            inputs={"Type":tc,"Air temperature [K]":air_temp,"Process temperature [K]":process_temp,"Rotational speed [rpm]":rpm,"Torque [Nm]":torque,"Tool wear [min]":tool_wear}
            row=pd.DataFrame([inputs])
            with st.spinner("Analyzing..."): time.sleep(0.4);features=engineer(row);prob=model.predict_proba(features)[0,1]
            status="Healthy" if prob<threshold else("Warning" if prob<0.8 else "Critical");add_to_history(inputs,prob,status)
            st.plotly_chart(make_gauge(prob),use_container_width=True)
            if prob>=0.8: st.markdown('<div class="status-critical"><h3>CRITICAL - IMMEDIATE MAINTENANCE REQUIRED</h3><p>Shut down machine and inspect immediately.</p></div>',unsafe_allow_html=True)
            elif prob>=threshold: st.markdown('<div class="status-danger"><h3>WARNING - SCHEDULE MAINTENANCE</h3><p>Plan inspection within 24 hours.</p></div>',unsafe_allow_html=True)
            else: st.markdown('<div class="status-healthy"><h3>OPERATIONAL - Machine Healthy</h3><p>All readings within normal range.</p></div>',unsafe_allow_html=True)
            st.markdown("");fd={col:features.iloc[0][col] for col in features.columns}
            t1,t2=st.tabs(["Sensor Levels","Radar View"])
            with t1: st.plotly_chart(make_feature_bars(fd),use_container_width=True)
            with t2: st.plotly_chart(make_radar(fd),use_container_width=True)
        else: st.markdown('<div style="text-align:center;padding:80px 20px;"><p style="font-size:4rem;margin:0;">🏭</p><h3 style="color:#888 !important;">Awaiting Sensor Input</h3><p style="color:#aaa;">Adjust readings and click <b>ANALYZE MACHINE</b>, or try a <b>Quick Preset</b></p></div>',unsafe_allow_html=True)

def page_batch():
    st.markdown('<div class="hero-banner"><h1>📊 Batch Machine Analysis</h1><p>Upload a CSV to analyze multiple machines at once</p></div>',unsafe_allow_html=True)
    st.markdown("Required columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`")
    st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Drop your sensor data CSV",type="csv")
    if uploaded:
        data=pd.read_csv(uploaded)
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            try:
                features=engineer(data);probs=model.predict_proba(features)[:,1];data["Failure Probability"]=probs
                data["Risk Level"]=pd.cut(probs,bins=[0,0.3,threshold,1.0],labels=["Low","Medium","High"])
                ar=int((probs>=threshold).sum());ca=int(((probs>=0.3)&(probs<threshold)).sum());lo=len(data)-ar-ca
                c1,c2,c3,c4=st.columns(4)
                c1.markdown(f'<div class="metric-card"><div class="metric-label">Total</div><div class="metric-value">{len(data)}</div></div>',unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-label">High Risk</div><div class="metric-value red">{ar}</div></div>',unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-label">Medium Risk</div><div class="metric-value yellow">{ca}</div></div>',unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-card"><div class="metric-label">Low Risk</div><div class="metric-value green">{lo}</div></div>',unsafe_allow_html=True)
                st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
                cc,cp=st.columns([3,1])
                with cc:
                    fig=go.Figure();fig.add_trace(go.Histogram(x=probs,nbinsx=50,marker_color="#ffc107",opacity=0.85));fig.add_vline(x=threshold,line_dash="dash",line_color="#c62828",annotation_text="Threshold")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",xaxis=dict(title="Failure Probability",gridcolor="#eee"),yaxis=dict(title="Count",gridcolor="#eee"),height=300,margin=dict(l=50,r=20,t=30,b=50))
                    st.plotly_chart(fig,use_container_width=True)
                with cp:
                    fig=go.Figure(data=[go.Pie(labels=["Low","Medium","High"],values=[lo,ca,ar],marker=dict(colors=["#4caf50","#ffc107","#ef5350"]),hole=0.5,textinfo="label+value")])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=300,margin=dict(l=10,r=10,t=10,b=10),showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)
                st.dataframe(data.sort_values("Failure Probability",ascending=False),use_container_width=True,height=400)
                st.download_button("DOWNLOAD RESULTS",data.to_csv(index=False).encode("utf-8"),"predictions.csv","text/csv",use_container_width=True)
            except Exception as e: st.error(f"Error: {e}");st.info("Check that your CSV has the required columns.")

def page_monitor():
    st.markdown('<div class="hero-banner"><h1>📈 Real-Time Monitoring</h1><p>Simulates a machine degrading over time with live failure predictions</p></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: speed=st.selectbox("Speed",["Fast (0.2s)","Normal (0.5s)","Slow (1.0s)"])
    with c2: n_steps=st.slider("Readings",20,100,40)
    with c3: degrade_rate=st.select_slider("Degradation Rate",options=["Slow","Medium","Fast"],value="Medium")
    sm={"Fast (0.2s)":0.2,"Normal (0.5s)":0.5,"Slow (1.0s)":1.0};dm={"Slow":0.5,"Medium":1.0,"Fast":2.0}
    st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
    start=st.button("START MONITORING",type="primary",use_container_width=True)
    if start:
        gc,cc=st.columns([1,2]);gp=gc.empty();cp=cc.empty();mp=st.empty();ap=st.empty();pp=st.empty()
        ph=[];al=0;dr=dm[degrade_rate];np.random.seed(None)
        for step in range(n_steps):
            d=(step/n_steps)*dr
            row=pd.DataFrame([{"Type":np.random.choice(["L","M","H"]),"Air temperature [K]":298+np.random.normal(2,0.5)+d*3,"Process temperature [K]":308+np.random.normal(1.5,0.3)+d*2,"Rotational speed [rpm]":max(1168,1500+np.random.normal(0,50)-d*200),"Torque [Nm]":min(76,max(4,40+np.random.normal(0,5)+d*20)),"Tool wear [min]":int(50+step*(200/n_steps))}])
            prob=model.predict_proba(engineer(row))[0,1];ph.append(prob)
            if prob>=threshold: al+=1
            gp.plotly_chart(make_gauge(prob),use_container_width=True)
            fig=go.Figure();fig.add_trace(go.Scatter(x=list(range(1,len(ph)+1)),y=ph,mode="lines+markers",line=dict(color="#ffc107",width=3),marker=dict(size=5,color=["#ef5350" if p>=threshold else "#ffc107" for p in ph]),fill="tozeroy",fillcolor="rgba(255,193,7,0.1)"))
            fig.add_hline(y=threshold,line_dash="dash",line_color="#c62828",annotation_text="Threshold")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",xaxis=dict(title="Reading #",gridcolor="#eee"),yaxis=dict(title="Probability",range=[0,1],gridcolor="#eee"),height=260,margin=dict(l=50,r=20,t=20,b=40),showlegend=False)
            cp.plotly_chart(fig,use_container_width=True)
            m1,m2,m3,m4=mp.columns(4);m1.metric("Current",f"{prob:.1%}");m2.metric("Maximum",f"{max(ph):.1%}");m3.metric("Average",f"{np.mean(ph):.1%}");m4.metric("Alerts",al)
            if prob>=threshold: ap.markdown(f'<div class="status-critical"><h3>ALERT - Reading #{step+1}: {prob:.1%} failure probability</h3></div>',unsafe_allow_html=True)
            elif prob>=0.3: ap.markdown(f'<div class="status-danger"><h3>CAUTION - Reading #{step+1}: {prob:.1%}</h3></div>',unsafe_allow_html=True)
            else: ap.empty()
            pp.progress((step+1)/n_steps,text=f"Reading {step+1} of {n_steps}");time.sleep(sm[speed])
        pp.empty();ap.markdown('<div class="status-healthy"><h3>Monitoring Session Complete</h3></div>',unsafe_allow_html=True)

def page_history():
    st.markdown('<div class="hero-banner"><h1>📜 Prediction History</h1><p>All predictions made during this session</p></div>',unsafe_allow_html=True)
    if not st.session_state.prediction_history: st.info("No predictions yet. Go to the Predict page to get started.");return
    recs=st.session_state.prediction_history;total=len(recs);alerts=sum(1 for r in recs if r["status"]!="Healthy");avg=np.mean([r["probability"] for r in recs])
    c1,c2,c3=st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Total Predictions</div><div class="metric-value">{total}</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Alerts</div><div class="metric-value red">{alerts}</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Avg Probability</div><div class="metric-value">{avg:.1%}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
    ps=[r["probability"] for r in reversed(recs)]
    fig=go.Figure();fig.add_trace(go.Scatter(y=ps,mode="lines+markers",line=dict(color="#ffc107",width=2),marker=dict(color=["#ef5350" if p>=threshold else "#4caf50" for p in ps],size=8),fill="tozeroy",fillcolor="rgba(255,193,7,0.1)"))
    fig.add_hline(y=threshold,line_dash="dash",line_color="#c62828")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",xaxis=dict(title="Prediction #",gridcolor="#eee"),yaxis=dict(title="Failure Probability",range=[0,1],gridcolor="#eee"),height=300,margin=dict(l=50,r=20,t=20,b=50))
    st.plotly_chart(fig,use_container_width=True)
    df=pd.DataFrame(recs);st.dataframe(df,use_container_width=True,height=400)
    st.download_button("DOWNLOAD HISTORY",df.to_csv(index=False).encode("utf-8"),"prediction_history.csv","text/csv",use_container_width=True)
    if st.button("CLEAR HISTORY",use_container_width=True): st.session_state.prediction_history=[];st.session_state.total_predictions=0;st.session_state.total_alerts=0;st.rerun()

def page_analytics():
    import os
    st.markdown('<div class="hero-banner"><h1>🧠 Model Analytics</h1><p>Understand how the model makes predictions using SHAP explainability</p></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Model Details")
        st.markdown("| Property | Value |\n|---|---|\n| **Algorithm** | Gradient Boosting |\n| **Dataset** | AI4I 2020 (10,000) |\n| **Features** | 6 raw + 5 engineered |\n| **Train/Test** | 80% / 20% |\n| **Tuning** | Optuna, 80 trials |")
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Performance")
        fig=go.Figure(go.Bar(x=[96.6,82.4,88.9,98.5],y=["Precision","Recall","F1","ROC-AUC"],orientation="h",marker=dict(color=["#4caf50","#ffc107","#2196f3","#9c27b0"]),text=["96.6%","82.4%","88.9%","98.5%"],textposition="outside"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fafafa",xaxis=dict(range=[0,110],gridcolor="#eee",title="Score (%)"),yaxis=dict(tickfont=dict(size=13)),height=220,margin=dict(l=80,r=40,t=10,b=40))
        st.plotly_chart(fig,use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True);st.subheader("SHAP Explainability Plots")
    sf={"Feature Importance (Summary)":"reports/09_shap_summary.png","Feature Importance (Bar)":"reports/10_shap_bar.png","Failure Explanation":"reports/11_shap_failure_waterfall.png","Healthy Explanation":"reports/12_shap_healthy_waterfall.png"}
    tabs=st.tabs(list(sf.keys()))
    for tab,(name,path) in zip(tabs,sf.items()):
        with tab:
            if os.path.exists(path): st.image(path,use_container_width=True)
            else: st.warning(f"Plot not found: {path}")
    st.markdown('<div class="hazard-stripe"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Engineered Features")
    st.dataframe(pd.DataFrame({"Feature":["temp_diff","power_w","strain","torque_per_rpm","temp_rpm_ratio"],"Formula":["Process-Air Temp","Torque*RPM*2pi/60","Wear*Torque","Torque/RPM","ProcTemp/RPM"],"Detects":["Heat dissipation","Power failure","Overstrain","Mechanical ratio","Overheating"]}),use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Confusion Matrix")
    cm=np.array([[1930,2],[12,56]])
    fig=go.Figure(data=go.Heatmap(z=cm,x=["Pred Healthy","Pred Failure"],y=["Act Healthy","Act Failure"],colorscale=[[0,"#fff8e1"],[1,"#ffc107"]],text=cm,texttemplate="%{text}",textfont=dict(size=20),showscale=False))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=300,margin=dict(l=120,r=20,t=20,b=60),yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig,use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

def page_settings():
    st.markdown('<div class="hero-banner"><h1>⚙️ Settings</h1><p>Configure your PredictX experience</p></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Profile")
        st.markdown(f"**Username:** {st.session_state.username}");st.markdown(f"**Role:** {USER_ROLES.get(st.session_state.username,'User')}")
        st.markdown(f"**Session Predictions:** {st.session_state.total_predictions}");st.markdown(f"**Session Alerts:** {st.session_state.total_alerts}")
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("Alert Settings")
        ct=st.slider("Custom Alert Threshold",0.1,0.9,float(threshold),0.05)
        if ct!=threshold: st.info(f"Custom threshold: {ct:.0%} (default: {threshold:.0%})")
        st.toggle("Email notifications (demo)",value=False);st.toggle("Sound alerts (demo)",value=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-card">',unsafe_allow_html=True);st.subheader("About PredictX")
    st.markdown("**PredictX** is an AI-powered predictive maintenance system.\n\n**Tech Stack:** Python, Scikit-learn, XGBoost, SHAP, Streamlit, Plotly")
    st.markdown('</div>',unsafe_allow_html=True)

if not st.session_state.logged_in: login_page()
else:
    render_sidebar()
    p=st.session_state.page
    if p=="Dashboard": page_dashboard()
    elif p=="Predict": page_predict()
    elif p=="Batch Analysis": page_batch()
    elif p=="Live Monitor": page_monitor()
    elif p=="History": page_history()
    elif p=="Analytics": page_analytics()
    elif p=="Settings": page_settings()
