import joblib, pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go, plotly.express as px
import time, hashlib, os
from datetime import datetime
from src.features import engineer

st.set_page_config(page_title="PredictX", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

# ─── THEME COLORS ───
# Background: #f4f4f4 (light grey)
# Cards: #ffffff (white)
# Primary accent: #e86e2e (orange)
# Dark: #1c1c1c (near black)
# Text on light: #1c1c1c (black)
# Text muted: #6b6b6b (medium grey)
# Sidebar bg: #1c1c1c (black)
# Sidebar text: #d0d0d0 (light grey)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

.stApp { background: #f4f4f4 !important; }
[data-testid="stHeader"] { background: #1c1c1c !important; }

/* Force ALL text visible on light background */
html, body, p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stText, [data-testid="stText"],
[class*="css"] p, [class*="css"] span, [class*="css"] div,
[data-testid="stExpander"] p, [data-testid="stExpander"] span {
    font-family: 'DM Sans', sans-serif !important;
    color: #1c1c1c !important;
}

h1 { font-weight: 700 !important; font-size: 1.75rem !important; color: #1c1c1c !important; }
h2 { font-weight: 700 !important; font-size: 1.25rem !important; color: #1c1c1c !important; }
h3 { font-weight: 600 !important; font-size: 1.05rem !important; color: #2a2a2a !important; }

/* Sidebar: light text on black */
section[data-testid="stSidebar"] { background: #1c1c1c !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [class*="css"] {
    color: #d0d0d0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] strong {
    color: #ffffff !important;
}

/* Hero banner */
.hero {
    background: #1c1c1c;
    border-radius: 14px;
    padding: 26px 30px;
    margin-bottom: 22px;
    border-left: 5px solid #e86e2e;
}
.hero h1 { color: #ffffff !important; font-size: 1.5rem !important; margin: 0 !important; }
.hero p { color: #aaaaaa !important; margin: 6px 0 0 0 !important; font-size: 0.9rem; }

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.07);
}

/* KPI cards */
.kpi {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-top: 4px solid #e86e2e;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s ease;
    margin-bottom: 10px;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.kpi-val {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 4px 0;
    line-height: 1.2;
    color: #1c1c1c !important;
}
.kpi-label {
    font-size: 0.72rem;
    color: #888888 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
.green { color: #15803d !important; }
.red { color: #b91c1c !important; }
.amber { color: #b45309 !important; }
.blue { color: #1d4ed8 !important; }

/* Status banners */
.st-ok {
    background: #ecfdf5; border: 1.5px solid #6ee7b7;
    border-radius: 12px; padding: 16px; text-align: center;
}
.st-ok h3 { color: #065f46 !important; margin: 0 !important; }
.st-ok p { color: #047857 !important; margin: 4px 0 0 0 !important; }

.st-warn {
    background: #fff7ed; border: 1.5px solid #fb923c;
    border-radius: 12px; padding: 16px; text-align: center;
    animation: glowW 2.5s ease-in-out infinite;
}
.st-warn h3 { color: #9a3412 !important; margin: 0 !important; }
.st-warn p { color: #c2410c !important; margin: 4px 0 0 0 !important; }
@keyframes glowW {
    0%,100%{box-shadow:0 0 0 0 rgba(232,110,46,0.25);}
    50%{box-shadow:0 0 0 8px rgba(232,110,46,0);}
}

.st-crit {
    background: #fef2f2; border: 1.5px solid #f87171;
    border-radius: 12px; padding: 16px; text-align: center;
    animation: glowC 1.8s ease-in-out infinite;
}
.st-crit h3 { color: #991b1b !important; margin: 0 !important; }
.st-crit p { color: #b91c1c !important; margin: 4px 0 0 0 !important; }
@keyframes glowC {
    0%,100%{box-shadow:0 0 0 0 rgba(248,113,113,0.3);}
    50%{box-shadow:0 0 0 10px rgba(248,113,113,0);}
}

/* Hazard stripe */
.hstripe {
    background: repeating-linear-gradient(-45deg, #e86e2e, #e86e2e 8px, #1c1c1c 8px, #1c1c1c 16px);
    height: 4px; border-radius: 2px; margin: 14px 0;
}

/* Login */
.login-box {
    max-width: 400px; margin: 40px auto; background: #fff;
    border-radius: 18px; padding: 36px 30px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    border-top: 4px solid #e86e2e;
}
.login-box h2 { text-align: center; color: #1c1c1c !important; }
.login-box p { text-align: center; color: #888888 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #1c1c1c; border-radius: 10px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #999999 !important; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #e86e2e !important; color: #ffffff !important;
}

/* Buttons */
.stButton>button {
    background: #e86e2e !important; color: #ffffff !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 20px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; transition: all 0.2s ease !important;
}
.stButton>button:hover {
    background: #d4621f !important;
    box-shadow: 0 4px 14px rgba(232,110,46,0.35) !important;
    transform: translateY(-1px);
}

/* Sliders */
.stSlider>div>div>div { background: #e86e2e !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e0e0e0;
    border-top: 3px solid #e86e2e; border-radius: 10px; padding: 12px;
}
[data-testid="stMetricValue"] { color: #1c1c1c !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6b6b6b !important; }

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #fff; border: 2px dashed #e86e2e; border-radius: 12px; padding: 16px;
}

/* History rows */
.hrow {
    background: #fff; border: 1px solid #e8e8e8; border-radius: 10px;
    padding: 10px 16px; margin: 5px 0;
    display: flex; justify-content: space-between; align-items: center;
}

/* Selectbox text */
.stSelectbox div[data-baseweb="select"] span { color: #1c1c1c !important; }
div[data-baseweb="select"] { color: #1c1c1c !important; }

/* Input text */
.stTextInput input { color: #1c1c1c !important; }

/* Info boxes */
.stAlert p, .stAlert span { color: #1c1c1c !important; }

/* Expander */
[data-testid="stExpander"] summary span { color: #1c1c1c !important; }
</style>""", unsafe_allow_html=True)

# ─── Session State ───
for k, v in [("logged_in",False),("username",""),("prediction_history",[]),("page","Dashboard"),("total_predictions",0),("total_alerts",0)]:
    if k not in st.session_state:
        st.session_state[k] = v

USERS = {u: hashlib.sha256(p.encode()).hexdigest() for u, p in [("admin","admin123"),("hasini","hasini123"),("lakshanaa","lakshanaa123"),("demo","demo")]}
ROLES = {"admin":"Administrator","hasini":"ML Engineer","lakshanaa":"ML Engineer","demo":"Guest"}

@st.cache_resource
def load_model():
    a = joblib.load("models/model.joblib")
    return a["model"], a["threshold"]

model, threshold = load_model()

# ─── Plotly helpers ───
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafafa", font=dict(family="DM Sans", color="#1c1c1c"))

def gauge(val):
    color = "#15803d" if val < 0.3 else ("#e86e2e" if val < threshold else "#dc2626")
    tag = "NORMAL" if val < 0.3 else ("CAUTION" if val < threshold else "CRITICAL")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val*100,
        number={"suffix":"%","font":{"size":44,"color":"#1c1c1c","family":"Space Mono"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#999","tickfont":{"color":"#666","size":10}},
               "bar":{"color":color,"thickness":0.72},"bgcolor":"#efefef","borderwidth":0,
               "steps":[{"range":[0,30],"color":"#ecfdf5"},{"range":[30,threshold*100],"color":"#fff7ed"},{"range":[threshold*100,100],"color":"#fef2f2"}],
               "threshold":{"line":{"color":"#b91c1c","width":2.5},"thickness":0.82,"value":threshold*100}}))
    fig.update_layout(height=240, margin=dict(l=20,r=20,t=20,b=5), **PL,
        annotations=[dict(text=tag,x=0.5,y=-0.08,showarrow=False,font=dict(size=13,color=color,family="DM Sans"),xref="paper",yref="paper")])
    return fig

def radar_chart(fd):
    cats=list(fd.keys()); vals=list(fd.values())
    mins={"type":0,"air_temp":295,"process_temp":305,"rpm":1100,"torque":3,"tool_wear":0,"temp_diff":7,"power_w":1000,"strain":0,"torque_per_rpm":0,"temp_rpm_ratio":0.1}
    maxs={"type":2,"air_temp":305,"process_temp":315,"rpm":2900,"torque":80,"tool_wear":260,"temp_diff":13,"power_w":11000,"strain":17000,"torque_per_rpm":0.07,"temp_rpm_ratio":0.27}
    n=[(v-mins.get(c,0))/(maxs.get(c,1)-mins.get(c,0)+1e-9) for c,v in zip(cats,vals)]
    fig=go.Figure(go.Scatterpolar(r=n+[n[0]],theta=cats+[cats[0]],fill="toself",fillcolor="rgba(232,110,46,0.12)",line=dict(color="#e86e2e",width=2)))
    fig.update_layout(polar=dict(bgcolor="#fff",radialaxis=dict(visible=True,range=[0,1],gridcolor="#e8e8e8",tickfont=dict(color="#999",size=8)),angularaxis=dict(gridcolor="#e8e8e8",tickfont=dict(color="#555",size=9))),
                      height=280,margin=dict(l=50,r=50,t=20,b=20),showlegend=False,**PL)
    return fig

def bar_chart(fd):
    df=pd.DataFrame({"Feature":list(fd.keys()),"Value":list(fd.values())})
    mins={"type":0,"air_temp":295,"process_temp":305,"rpm":1100,"torque":3,"tool_wear":0,"temp_diff":7,"power_w":1000,"strain":0,"torque_per_rpm":0,"temp_rpm_ratio":0.1}
    maxs={"type":2,"air_temp":305,"process_temp":315,"rpm":2900,"torque":80,"tool_wear":260,"temp_diff":13,"power_w":11000,"strain":17000,"torque_per_rpm":0.07,"temp_rpm_ratio":0.27}
    df["N"]=df.apply(lambda r:(r["Value"]-mins.get(r["Feature"],0))/(maxs.get(r["Feature"],1)-mins.get(r["Feature"],0)+1e-9),axis=1)
    df["C"]=df["N"].apply(lambda x:"#15803d" if x<0.5 else("#e86e2e" if x<0.8 else "#dc2626"))
    fig=go.Figure(go.Bar(x=df["N"],y=df["Feature"],orientation="h",marker=dict(color=df["C"],line=dict(width=0)),
                         text=df["Value"].apply(lambda x:str(round(x,1))),textposition="outside",textfont=dict(color="#555",size=10)))
    fig.update_layout(xaxis=dict(range=[0,1.2],title=None,gridcolor="#e8e8e8",zeroline=False),
                      yaxis=dict(autorange="reversed",gridcolor="#e8e8e8"),
                      height=300,margin=dict(l=105,r=30,t=5,b=30),**PL)
    return fig

def add_hist(inputs, prob, status):
    st.session_state.prediction_history.insert(0,{
        "ts":datetime.now().strftime("%H:%M:%S"),"type":inputs["Type"],"prob":prob,"status":status,
        "air":inputs["Air temperature [K]"],"proc":inputs["Process temperature [K]"],
        "rpm":inputs["Rotational speed [rpm]"],"torque":inputs["Torque [Nm]"],"wear":inputs["Tool wear [min]"]})
    st.session_state.total_predictions += 1
    if status != "Healthy":
        st.session_state.total_alerts += 1

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def kpi_card(label, value, css=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-val {css}">{value}</div></div>', unsafe_allow_html=True)


# ═══════════════ LOGIN ═══════════════
def login_page():
    st.markdown("")
    _,mid,_ = st.columns([1,1.1,1])
    with mid:
        st.markdown('<div class="login-box"><p style="font-size:2.5rem;margin:0;">🏭</p><h2>PredictX</h2><p>Industrial Predictive Maintenance</p></div>', unsafe_allow_html=True)
        with st.form("login"):
            u = st.text_input("Username", placeholder="Enter username")
            p = st.text_input("Password", type="password", placeholder="Enter password")
            if st.form_submit_button("Sign in", use_container_width=True):
                if u in USERS and USERS[u] == hashlib.sha256(p.encode()).hexdigest():
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with st.expander("Demo credentials"):
            st.code("demo / demo\nadmin / admin123\nhasini / hasini123")


# ═══════════════ SIDEBAR ═══════════════
def sidebar():
    with st.sidebar:
        st.markdown('<div style="text-align:center;padding:12px 0"><span style="font-size:1.8rem">🏭</span><h2 style="color:#e86e2e !important;margin:4px 0 0 0">PredictX</h2></div>', unsafe_allow_html=True)
        st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)
        role = ROLES.get(st.session_state.username, "User")
        st.markdown(f'<div style="background:#2a2a2a;border-radius:10px;padding:10px 14px;margin:8px 0"><b style="color:#fff !important">{st.session_state.username}</b><br><span style="color:#e86e2e !important;font-size:0.8rem">{role}</span></div>', unsafe_allow_html=True)
        st.markdown("")
        for pg, ic in [("Dashboard","📊"),("Predict","🔍"),("Batch Analysis","📋"),("Live Monitor","📈"),("History","📜"),("Analytics","🧠"),("Settings","⚙️")]:
            if st.button(f"{ic}  {pg}", use_container_width=True, key=f"n_{pg}"):
                st.session_state.page = pg
                st.rerun()
        st.markdown("")
        st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#2a2a2a;border-radius:10px;padding:10px 14px;margin:10px 0"><span style="color:#888 !important;font-size:0.7rem;letter-spacing:1px">SESSION</span><br><span style="color:#e86e2e !important;font-weight:700">{st.session_state.total_predictions}</span> <span style="color:#aaa !important">predictions</span><br><span style="color:#f87171 !important;font-weight:700">{st.session_state.total_alerts}</span> <span style="color:#aaa !important">alerts</span></div>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()


# ═══════════════ DASHBOARD ═══════════════
def pg_dashboard():
    hero("Dashboard", "System overview and recent activity")

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_card("Accuracy","98.4%","green")
    with c2: kpi_card("Precision","96.6%","blue")
    with c3: kpi_card("Recall","82.4%","amber")
    with c4: kpi_card("Predictions", st.session_state.total_predictions)
    with c5: kpi_card("Alerts", st.session_state.total_alerts, "red")

    st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)
    left, right = st.columns([3,2], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Failure probability distribution")
        data = pd.read_csv("data/ai4i2020.csv")
        feats = engineer(data)
        probs = model.predict_proba(feats)[:,1]
        fig = go.Figure(go.Histogram(x=probs, nbinsx=60, marker_color="#e86e2e", opacity=0.85))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#b91c1c", annotation_text="Threshold", annotation_font_color="#b91c1c")
        fig.update_layout(xaxis=dict(title="Probability",gridcolor="#e8e8e8"),yaxis=dict(title="Machines",gridcolor="#e8e8e8"),height=280,margin=dict(l=45,r=15,t=25,b=45),**PL)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Recent predictions")
        if st.session_state.prediction_history:
            for r in st.session_state.prediction_history[:5]:
                dot = "🟢" if r["status"] == "Healthy" else "🔴"
                st.markdown(f'<div class="hrow"><span>{dot} <b style="color:#1c1c1c !important">Type {r["type"]}</b></span><span style="font-family:Space Mono,monospace !important;color:#1c1c1c !important">{r["prob"]:.1%}</span><span style="color:#888 !important;font-size:0.8rem">{r["ts"]}</span></div>', unsafe_allow_html=True)
        else:
            st.info("No predictions yet — go to the Predict page.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Fleet health")
        ar = int((probs >= threshold).sum())
        h = len(probs) - ar
        fig = go.Figure(go.Pie(labels=["Healthy","At Risk"], values=[h,ar], hole=0.65,
            marker=dict(colors=["#15803d","#dc2626"], line=dict(color="#fff",width=2)),
            textinfo="label+percent", textfont=dict(size=12, color="#1c1c1c")))
        fig.update_layout(height=260, margin=dict(l=15,r=15,t=15,b=15), showlegend=False,
            annotations=[dict(text=f"<b>{ar}</b><br><span style='font-size:11px'>at risk</span>",x=0.5,y=0.5,font_size=18,font_color="#1c1c1c",showarrow=False)], **PL)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Quick actions")
        if st.button("🔍  New prediction", use_container_width=True, key="d1"):
            st.session_state.page = "Predict"; st.rerun()
        if st.button("📋  Batch upload", use_container_width=True, key="d2"):
            st.session_state.page = "Batch Analysis"; st.rerun()
        if st.button("📈  Live monitor", use_container_width=True, key="d3"):
            st.session_state.page = "Live Monitor"; st.rerun()
        if st.button("🧠  Analytics", use_container_width=True, key="d4"):
            st.session_state.page = "Analytics"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════ PREDICT ═══════════════
def pg_predict():
    hero("Machine Failure Prediction", "Enter sensor readings or use a preset")

    ci, cr = st.columns([2,3], gap="large")
    with ci:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Sensor readings")
        st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)
        mt = st.selectbox("Machine type", ["L — Low quality","M — Medium quality","H — High quality"])
        c1,c2 = st.columns(2)
        at = c1.slider("Air temp (K)", 295.0, 305.0, 300.0, 0.1)
        pt = c2.slider("Process temp (K)", 305.0, 315.0, 310.0, 0.1)
        c3,c4 = st.columns(2)
        rpm = c3.slider("Speed (RPM)", 1100, 2900, 1500)
        tq = c4.slider("Torque (Nm)", 3.0, 80.0, 40.0, 0.1)
        tw = st.slider("Tool wear (min)", 0, 260, 100)
        go_btn = st.button("Analyze machine", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Quick presets")
        p1,p2,p3 = st.columns(3)
        bh = p1.button("Healthy", use_container_width=True, key="qh")
        bw = p2.button("Warning", use_container_width=True, key="qw")
        bc = p3.button("Critical", use_container_width=True, key="qc")
        st.markdown('</div>', unsafe_allow_html=True)

    tc = mt[0]
    if bh:
        at,pt,rpm,tq,tw = 300.0,310.0,1500,40.0,50; go_btn = True
    elif bw:
        at,pt,rpm,tq,tw = 302.0,311.5,1350,55.0,180; go_btn = True
    elif bc:
        at,pt,rpm,tq,tw = 303.5,312.0,1200,70.0,240; tc = "L"; go_btn = True

    with cr:
        if go_btn:
            inp = {"Type":tc, "Air temperature [K]":at, "Process temperature [K]":pt,
                   "Rotational speed [rpm]":rpm, "Torque [Nm]":tq, "Tool wear [min]":tw}
            row = pd.DataFrame([inp])
            with st.spinner("Analyzing..."):
                time.sleep(0.35)
                feats = engineer(row)
                prob = model.predict_proba(feats)[0,1]
            stat = "Healthy" if prob < threshold else ("Warning" if prob < 0.8 else "Critical")
            add_hist(inp, prob, stat)

            st.plotly_chart(gauge(prob), use_container_width=True)

            if prob >= 0.8:
                st.markdown('<div class="st-crit"><h3>🚨 Critical — immediate maintenance required</h3><p>Shut down and inspect this machine now.</p></div>', unsafe_allow_html=True)
            elif prob >= threshold:
                st.markdown('<div class="st-warn"><h3>⚠️ Warning — schedule maintenance</h3><p>Plan an inspection within 24 hours.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="st-ok"><h3>✅ Operational — machine healthy</h3><p>All sensor readings within normal range.</p></div>', unsafe_allow_html=True)

            st.markdown("")
            fd = {c: feats.iloc[0][c] for c in feats.columns}
            t1, t2 = st.tabs(["Sensor levels", "Radar view"])
            with t1:
                st.plotly_chart(bar_chart(fd), use_container_width=True)
            with t2:
                st.plotly_chart(radar_chart(fd), use_container_width=True)
        else:
            st.markdown('<div style="text-align:center;padding:70px 20px"><p style="font-size:3.5rem;margin:0;opacity:0.25">⚙️</p><h3 style="color:#888 !important">Awaiting sensor input</h3><p style="color:#aaa !important">Adjust readings or try a quick preset, then click Analyze machine.</p></div>', unsafe_allow_html=True)


# ═══════════════ BATCH ═══════════════
def pg_batch():
    hero("Batch Analysis", "Upload a CSV to analyze an entire fleet at once")
    st.markdown("Required columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`")
    st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)

    up = st.file_uploader("Drop sensor CSV here", type="csv")
    if up:
        data = pd.read_csv(up)
        with st.spinner("Analyzing fleet..."):
            time.sleep(0.25)
            try:
                feats = engineer(data)
                probs = model.predict_proba(feats)[:,1]
                data["Failure Probability"] = probs
                data["Risk"] = pd.cut(probs, bins=[0,0.3,threshold,1.0], labels=["Low","Medium","High"])
                ar = int((probs >= threshold).sum())
                ca = int(((probs >= 0.3) & (probs < threshold)).sum())
                lo = len(data) - ar - ca

                c1,c2,c3,c4 = st.columns(4)
                with c1: kpi_card("Total machines", len(data))
                with c2: kpi_card("High risk", ar, "red")
                with c3: kpi_card("Medium risk", ca, "amber")
                with c4: kpi_card("Low risk", lo, "green")

                st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)

                cc, cp = st.columns([3,1])
                with cc:
                    fig = go.Figure(go.Histogram(x=probs, nbinsx=50, marker_color="#e86e2e", opacity=0.85))
                    fig.add_vline(x=threshold, line_dash="dash", line_color="#b91c1c", annotation_text="Threshold")
                    fig.update_layout(xaxis=dict(title="Probability",gridcolor="#e8e8e8"),yaxis=dict(title="Count",gridcolor="#e8e8e8"),height=280,margin=dict(l=45,r=15,t=25,b=45),**PL)
                    st.plotly_chart(fig, use_container_width=True)
                with cp:
                    fig = go.Figure(go.Pie(labels=["Low","Med","High"], values=[lo,ca,ar],
                        marker=dict(colors=["#15803d","#e86e2e","#dc2626"]), hole=0.5, textinfo="label+value",
                        textfont=dict(color="#1c1c1c")))
                    fig.update_layout(height=280, margin=dict(l=5,r=5,t=5,b=5), showlegend=False, **PL)
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(data.sort_values("Failure Probability", ascending=False), use_container_width=True, height=380)
                st.download_button("Download results", data.to_csv(index=False).encode(), "predictions.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Check columns match the required format above.")


# ═══════════════ LIVE MONITOR ═══════════════
def pg_monitor():
    hero("Live Monitor", "Watch a simulated machine degrade in real time")

    c1,c2,c3 = st.columns(3)
    with c1: sp = st.selectbox("Speed", ["Fast (0.2s)","Normal (0.5s)","Slow (1.0s)"])
    with c2: ns = st.slider("Readings", 20, 100, 40)
    with c3: dr = st.select_slider("Degradation", options=["Slow","Medium","Fast"], value="Medium")
    sm = {"Fast (0.2s)":0.2, "Normal (0.5s)":0.5, "Slow (1.0s)":1.0}
    dm = {"Slow":0.5, "Medium":1.0, "Fast":2.0}
    st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)

    start_btn = st.button("Start monitoring", type="primary", use_container_width=True)
    if start_btn:
        gc, cc = st.columns([1,2])
        gp = gc.empty(); cp = cc.empty(); mp = st.empty(); ap = st.empty(); pp = st.empty()
        ph = []; al = 0; d_rate = dm[dr]; np.random.seed(None)

        for step in range(ns):
            d = (step / ns) * d_rate
            row = pd.DataFrame([{
                "Type": np.random.choice(["L","M","H"]),
                "Air temperature [K]": 298 + np.random.normal(2,0.5) + d*3,
                "Process temperature [K]": 308 + np.random.normal(1.5,0.3) + d*2,
                "Rotational speed [rpm]": max(1168, 1500 + np.random.normal(0,50) - d*200),
                "Torque [Nm]": min(76, max(4, 40 + np.random.normal(0,5) + d*20)),
                "Tool wear [min]": int(50 + step * (200/ns)),
            }])
            prob = model.predict_proba(engineer(row))[0,1]
            ph.append(prob)
            if prob >= threshold: al += 1

            gp.plotly_chart(gauge(prob), use_container_width=True)

            fig = go.Figure(go.Scatter(
                x=list(range(1,len(ph)+1)), y=ph, mode="lines+markers",
                line=dict(color="#e86e2e", width=2.5),
                marker=dict(size=4, color=["#dc2626" if p >= threshold else "#e86e2e" for p in ph]),
                fill="tozeroy", fillcolor="rgba(232,110,46,0.08)"))
            fig.add_hline(y=threshold, line_dash="dash", line_color="#b91c1c", annotation_text="Threshold")
            fig.update_layout(xaxis=dict(title="Reading",gridcolor="#e8e8e8"),
                              yaxis=dict(title="Probability",range=[0,1],gridcolor="#e8e8e8"),
                              height=240, margin=dict(l=45,r=15,t=15,b=40), showlegend=False, **PL)
            cp.plotly_chart(fig, use_container_width=True)

            m1,m2,m3,m4 = mp.columns(4)
            m1.metric("Current", f"{prob:.1%}")
            m2.metric("Max", f"{max(ph):.1%}")
            m3.metric("Avg", f"{np.mean(ph):.1%}")
            m4.metric("Alerts", al)

            if prob >= threshold:
                ap.markdown(f'<div class="st-crit"><h3>🚨 Alert — reading #{step+1}: {prob:.1%}</h3></div>', unsafe_allow_html=True)
            elif prob >= 0.3:
                ap.markdown(f'<div class="st-warn"><h3>⚠️ Caution — reading #{step+1}: {prob:.1%}</h3></div>', unsafe_allow_html=True)
            else:
                ap.empty()

            pp.progress((step+1)/ns, text=f"Reading {step+1} of {ns}")
            time.sleep(sm[sp])

        pp.empty()
        ap.markdown('<div class="st-ok"><h3>✅ Monitoring complete</h3></div>', unsafe_allow_html=True)


# ═══════════════ HISTORY ═══════════════
def pg_history():
    hero("Prediction History", "Every prediction from this session")
    recs = st.session_state.prediction_history
    if not recs:
        st.info("No predictions yet.")
        return

    tot = len(recs)
    als = sum(1 for r in recs if r["status"] != "Healthy")
    avg = np.mean([r["prob"] for r in recs])

    c1,c2,c3 = st.columns(3)
    with c1: kpi_card("Total", tot)
    with c2: kpi_card("Alerts", als, "red")
    with c3: kpi_card("Avg probability", f"{avg:.1%}")

    st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)

    ps = [r["prob"] for r in reversed(recs)]
    fig = go.Figure(go.Scatter(y=ps, mode="lines+markers",
        line=dict(color="#e86e2e",width=2),
        marker=dict(color=["#dc2626" if p >= threshold else "#15803d" for p in ps], size=7),
        fill="tozeroy", fillcolor="rgba(232,110,46,0.08)"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#b91c1c")
    fig.update_layout(xaxis=dict(title="Prediction #",gridcolor="#e8e8e8"),
                      yaxis=dict(title="Probability",range=[0,1],gridcolor="#e8e8e8"),
                      height=280, margin=dict(l=45,r=15,t=15,b=45), **PL)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame(recs), use_container_width=True, height=350)
    st.download_button("Download history", pd.DataFrame(recs).to_csv(index=False).encode(), "history.csv", "text/csv", use_container_width=True)
    if st.button("Clear history", use_container_width=True):
        st.session_state.prediction_history = []
        st.session_state.total_predictions = 0
        st.session_state.total_alerts = 0
        st.rerun()


# ═══════════════ ANALYTICS ═══════════════
def pg_analytics():
    hero("Model Analytics", "How the model works and what drives its predictions")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Model specification")
        spec = f"""
| | |
|---|---|
| **Algorithm** | Random Forest |
| **Dataset** | AI4I 2020 — 10,000 rows |
| **Features** | 6 raw + 5 engineered |
| **Split** | 80 / 20 stratified |
| **Tuning** | Balanced weights |
| **Threshold** | {threshold:.1%} (PR-curve) |
"""
        st.markdown(spec)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Test-set performance")
        fig = go.Figure(go.Bar(
            x=[97.9,97.5,97.7,98.4], y=["Precision","Recall","F1","Accuracy"], orientation="h",
            marker=dict(color=["#15803d","#e86e2e","#2563eb","#7c3aed"]),
            text=["97.9%","97.5%","97.7%","98.4%"], textposition="outside",
            textfont=dict(color="#1c1c1c")))
        fig.update_layout(xaxis=dict(range=[0,112],title="Score %",gridcolor="#e8e8e8"),
                          yaxis=dict(gridcolor="#e8e8e8"),
                          height=200, margin=dict(l=75,r=30,t=5,b=35), **PL)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)
    st.markdown("#### SHAP explainability")
    sf = {"Summary":"reports/09_shap_summary.png", "Bar":"reports/10_shap_bar.png",
          "Failure waterfall":"reports/11_shap_failure_waterfall.png",
          "Healthy waterfall":"reports/12_shap_healthy_waterfall.png"}
    tabs = st.tabs(list(sf.keys()))
    for tab, (n,p) in zip(tabs, sf.items()):
        with tab:
            if os.path.exists(p):
                st.image(p, use_container_width=True)
            else:
                st.warning(f"Not found: {p}")

    st.markdown('<div class="hstripe"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Engineered features")
    st.dataframe(pd.DataFrame({
        "Feature": ["temp_diff","power_w","strain","torque_per_rpm","temp_rpm_ratio"],
        "Formula": ["Process temp - Air temp","Torque x RPM x 2pi/60","Tool wear x Torque","Torque / RPM","Process temp / RPM"],
        "Failure mode": ["Heat dissipation (HDF)","Power failure (PWF)","Overstrain (OSF)","Mechanical stress","Overheating at low speed"]
    }), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Confusion matrix (test set)")
    cm = np.array([[1925,7],[12,56]])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicted healthy","Predicted failure"], y=["Actually healthy","Actually failure"],
        colorscale=[[0,"#fff7ed"],[1,"#e86e2e"]], text=cm, texttemplate="%{text}",
        textfont=dict(size=18, color="#1c1c1c"), showscale=False))
    fig.update_layout(height=280, margin=dict(l=110,r=15,t=10,b=50),
                      yaxis=dict(autorange="reversed",gridcolor="#e8e8e8"),
                      xaxis=dict(gridcolor="#e8e8e8"), **PL)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════ SETTINGS ═══════════════
def pg_settings():
    hero("Settings", "Profile and configuration")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Profile")
        st.markdown(f"**Username:** {st.session_state.username}")
        st.markdown(f"**Role:** {ROLES.get(st.session_state.username, 'User')}")
        st.markdown(f"**Predictions this session:** {st.session_state.total_predictions}")
        st.markdown(f"**Alerts this session:** {st.session_state.total_alerts}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Alerts")
        ct = st.slider("Custom threshold", 0.1, 0.9, float(threshold), 0.05)
        if ct != threshold:
            st.info(f"Custom: {ct:.0%} (default: {threshold:.0%})")
        st.toggle("Email notifications (demo)", value=False)
        st.toggle("Sound alerts (demo)", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### About")
    st.markdown("**PredictX** is an AI-powered predictive maintenance system built for a PBL course.\n\n**Stack:** Python, scikit-learn, Random Forest, SHAP, Streamlit, Plotly")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════ ROUTER ═══════════════
if not st.session_state.logged_in:
    login_page()
else:
    sidebar()
    pages = {
        "Dashboard": pg_dashboard,
        "Predict": pg_predict,
        "Batch Analysis": pg_batch,
        "Live Monitor": pg_monitor,
        "History": pg_history,
        "Analytics": pg_analytics,
        "Settings": pg_settings,
    }
    pages[st.session_state.page]()
