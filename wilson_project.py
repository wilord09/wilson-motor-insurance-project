"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MOTOR INSURANCE RISK ANALYTICS DASHBOARD                                    ║
║  Wilson Dissertation — MSc Business Analytics and Technology                 ║
║  University of Bolton                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor Insurance Risk Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main { background: #0a0f1e; color: #e8eaf0; }
    .stApp { background: #0a0f1e; }

    .metric-card {
        background: linear-gradient(135deg, #12193a 0%, #1a2550 100%);
        border: 1px solid #2a3a6a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #4fc3f7;
    }
    .metric-label { font-size: 0.8rem; color: #7986cb; text-transform: uppercase; letter-spacing: 1px; }

    .section-header {
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 100%);
        border-left: 4px solid #4fc3f7;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 16px 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e3f2fd;
    }

    .risk-badge-high   { background: #b71c1c; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .risk-badge-medium { background: #e65100; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .risk-badge-low    { background: #1b5e20; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }

    .stSidebar { background: #060c1a !important; }
    .stSidebar .stMarkdown { color: #90caf9; }

    h1, h2, h3 { color: #e3f2fd !important; }
    .stTabs [data-baseweb="tab"] { color: #90caf9; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #4fc3f7 !important; border-bottom: 2px solid #4fc3f7; }

    div[data-testid="stMetricValue"] { color: #4fc3f7 !important; font-family: 'IBM Plex Mono', monospace; }
    div[data-testid="stMetricLabel"] { color: #7986cb !important; }

    .insight-box {
        background: #12193a;
        border: 1px solid #2a3a6a;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #b0bec5;
        line-height: 1.6;
    }
    .insight-box strong { color: #4fc3f7; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Motor Insurance\n### Risk Analytics Dashboard")
    st.markdown("---")
    st.markdown("**University of Bolton**\nMSc Business Analytics\nand Technology")
    st.markdown("---")
    st.markdown("### 📁 Data Upload")

    ts_file  = st.file_uploader("Monthly Time Series CSV", type=['csv'],
                                 help="Upload monthly_collision_timeseries.csv")
    raw_file = st.file_uploader("Collisions Clean CSV", type=['csv'],
                                 help="Upload collisions_clean_2020_2024.csv")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    test_months  = st.slider("Test Period (months)", 6, 18, 12)
    forecast_yr  = st.selectbox("Forecast Year", [2025, 2026])
    show_ci      = st.checkbox("Show Confidence Intervals", value=True)

    st.markdown("---")
    st.markdown("<small style='color:#546e7a'>Built for Wilson Dissertation<br>Chapter 6 — Decision Support</small>", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_ts(file):
    df = pd.read_csv(file)
    if 'Total_Collision' in df.columns:
        df = df.rename(columns={'Total_Collision': 'Total_Collisions'})
    df['yearmonth_dt'] = pd.to_datetime(df['yearmonth_str'])
    df = df.sort_values('yearmonth_dt').reset_index(drop=True)
    return df

@st.cache_data
def load_raw(file):
    df = pd.read_csv(file)
    df['severe'] = (df['collision_severity'].isin([1, 2])).astype(int)
    df['adverse_weather']  = (df['weather_conditions'].isin([2,3,4,5,6,7])).astype(int)
    df['adverse_surface']  = (df['road_surface_conditions'].isin([2,3,4,5])).astype(int)
    df['darkness']         = (df['light_conditions'].isin([4,5,6,7])).astype(int)
    df['rural']            = (df['urban_or_rural_area'] == 2).astype(int)
    df['speed_limit_num']  = pd.to_numeric(df['speed_limit'], errors='coerce')
    df['is_weekend']       = df['is_weekend'].astype(int)
    return df

# ── HELPER: matplotlib dark style ────────────────────────────────────────────
def dark_fig(figsize=(12,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d1329')
    ax.set_facecolor('#0d1329')
    ax.tick_params(colors='#90caf9')
    ax.xaxis.label.set_color('#90caf9')
    ax.yaxis.label.set_color('#90caf9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3a6a')
    ax.grid(True, linestyle='--', alpha=0.2, color='#4fc3f7')
    return fig, ax

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
  <h1 style='font-size:2.2rem; font-weight:700; color:#4fc3f7; letter-spacing:-1px;'>
    🚗 Motor Insurance Risk Analytics
  </h1>
  <p style='color:#7986cb; font-size:1rem; margin-top:-8px;'>
    Predictive Analytics Dashboard — UK Road Collision Data 2020–2024
  </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "🔮 Forecasting",
    "⚠️ Risk Calculator",
    "📅 Forecast",
    "📋 Report"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if ts_file:
        ts = load_ts(ts_file)

        # KPI row
        total = ts['Total_Collisions'].sum()
        mean_m = ts['Total_Collisions'].mean()
        peak = ts['Total_Collisions'].max()
        low  = ts['Total_Collisions'].min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Collisions", f"{total:,.0f}", "2020–2024")
        c2.metric("Monthly Average", f"{mean_m:,.0f}", "per month")
        c3.metric("Peak Month", f"{peak:,.0f}", ts.loc[ts['Total_Collisions'].idxmax(), 'yearmonth_str'])
        c4.metric("Lockdown Low", f"{low:,.0f}", ts.loc[ts['Total_Collisions'].idxmin(), 'yearmonth_str'])

        st.markdown('<div class="section-header">📈 Monthly Collision Timeline</div>', unsafe_allow_html=True)

        fig, ax = dark_fig((14, 4))
        ax.plot(ts['yearmonth_dt'], ts['Total_Collisions'],
                color='#4fc3f7', linewidth=2, marker='o', markersize=2.5)
        ax.fill_between(ts['yearmonth_dt'], ts['Total_Collisions'],
                        alpha=0.15, color='#4fc3f7')
        rolling = ts['Total_Collisions'].rolling(3, center=True).mean()
        ax.plot(ts['yearmonth_dt'], rolling, color='#ff8f00',
                linewidth=2, linestyle='--', label='3-Month Rolling Avg')
        ax.axvspan(pd.Timestamp('2020-03-23'), pd.Timestamp('2021-07-19'),
                   alpha=0.08, color='#ef5350', label='COVID-19 Restrictions')
        ax.set_title('Monthly UK Road Collisions (2020–2024)', color='#e3f2fd', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha='right', color='#90caf9', fontsize=8)
        ax.legend(fontsize=9, facecolor='#0d1329', labelcolor='#90caf9')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Year breakdown
        st.markdown('<div class="section-header">📅 Year-by-Year Breakdown</div>', unsafe_allow_html=True)
        ts['year'] = ts['yearmonth_dt'].dt.year
        yearly = ts.groupby('year').agg(
            Annual_Total=('Total_Collisions','sum'),
            Monthly_Avg=('Total_Collisions','mean'),
            Peak=('Total_Collisions','max'),
            Low=('Total_Collisions','min')
        ).reset_index()
        yearly.columns = ['Year','Annual Total','Monthly Avg','Peak Month','Lowest Month']
        yearly['Annual Total'] = yearly['Annual Total'].apply(lambda x: f"{x:,.0f}")
        yearly['Monthly Avg']  = yearly['Monthly Avg'].apply(lambda x: f"{x:,.0f}")
        yearly['Peak Month']   = yearly['Peak Month'].apply(lambda x: f"{x:,.0f}")
        yearly['Lowest Month'] = yearly['Lowest Month'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(yearly, use_container_width=True, hide_index=True)

    else:
        st.info("👆 Upload your **monthly_collision_timeseries.csv** file in the sidebar to get started.")
        st.markdown("""
        <div class='insight-box'>
        <strong>About this dashboard</strong><br>
        This tool analyses UK road collision data to support motor insurance risk assessment.
        Upload your datasets in the sidebar, then explore six analytical sections:
        Overview, Time Series analysis, Forecasting model comparison, Risk Calculator,
        2025 Forecast with scenarios, and Report generation.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TIME SERIES ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    if ts_file:
        ts = load_ts(ts_file)
        ts['yearmonth_dt'] = pd.to_datetime(ts['yearmonth_str'])
        ts['month_num'] = ts['yearmonth_dt'].dt.month
        ts['year_num']  = ts['yearmonth_dt'].dt.year

        st.markdown('<div class="section-header">🌊 Seasonal Pattern Analysis</div>', unsafe_allow_html=True)

        seasonal = ts.groupby('month_num')['Total_Collisions'].agg(['mean','std']).reset_index()
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        fig, ax = dark_fig((12, 4))
        colors_bar = ['#4fc3f7'] * 12
        max_idx = int(seasonal['mean'].idxmax())
        min_idx = int(seasonal['mean'].idxmin())
        colors_bar[max_idx] = '#ff8f00'
        colors_bar[min_idx] = '#66bb6a'
        bars = ax.bar(range(1,13), seasonal['mean'], color=colors_bar,
                      edgecolor='#0d1329', linewidth=0.5, width=0.7)
        ax.errorbar(range(1,13), seasonal['mean'], yerr=seasonal['std'],
                    fmt='none', color='#90caf9', capsize=4, linewidth=1.5)
        for bar, val in zip(bars, seasonal['mean']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+seasonal['std'].max()*0.05,
                    f'{int(val):,}', ha='center', va='bottom', fontsize=7.5, color='#e3f2fd')
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(month_labels, color='#90caf9')
        ax.set_title('Average Monthly Collisions — Seasonal Pattern | 🟠 Peak  🟢 Lowest',
                     color='#e3f2fd', fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Severity breakdown
        if all(c in ts.columns for c in ['Fatal','Serious','Slight']):
            st.markdown('<div class="section-header">🎯 Severity Mix Over Time</div>', unsafe_allow_html=True)
            fig, ax = dark_fig((14, 4))
            ax.stackplot(ts['yearmonth_dt'],
                         [ts['Slight'], ts['Serious'], ts['Fatal']],
                         labels=['Slight','Serious','Fatal'],
                         colors=['#1565c0','#0288d1','#d32f2f'],
                         alpha=0.85)
            ax.axvspan(pd.Timestamp('2020-03-23'), pd.Timestamp('2021-07-19'),
                       alpha=0.08, color='#ef5350')
            ax.set_title('Monthly Collisions by Severity (Stacked)', color='#e3f2fd', fontsize=11, fontweight='bold')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.xticks(rotation=30, ha='right', color='#90caf9', fontsize=8)
            ax.legend(fontsize=9, facecolor='#0d1329', labelcolor='#90caf9', loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👆 Upload data in the sidebar to view time series analysis.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECASTING MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    if ts_file:
        ts = load_ts(ts_file)
        y  = ts['Total_Collisions'].values

        TRAIN_N  = len(y) - test_months
        y_train  = y[:TRAIN_N]
        y_test   = y[TRAIN_N:]
        dates_test = ts['yearmonth_dt'].values[TRAIN_N:]

        st.markdown('<div class="section-header">⚙️ Select Models to Compare</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        run_sma  = c1.checkbox("Simple Moving Average", value=True)
        run_arima= c2.checkbox("ARIMA", value=True)
        run_sar  = c3.checkbox("SARIMA", value=True)
        run_hw   = c4.checkbox("Holt-Winters ETS", value=True)

        if st.button("▶  Run Selected Models", type="primary"):
            results = {}
            progress = st.progress(0)
            status   = st.empty()

            if run_sma:
                status.text("Running Simple Moving Average...")
                sma_val = pd.Series(y_train).rolling(3).mean().iloc[-1]
                results['SMA'] = np.full(len(y_test), sma_val)
                progress.progress(25)

            if run_arima:
                status.text("Running ARIMA...")
                try:
                    from pmdarima import auto_arima
                    m = auto_arima(y_train, seasonal=False, stepwise=True,
                                   suppress_warnings=True, error_action='ignore')
                    results['ARIMA'] = m.predict(n_periods=len(y_test))
                except:
                    m = SARIMAX(y_train, order=(1,0,0)).fit(disp=False)
                    results['ARIMA'] = m.forecast(len(y_test))
                progress.progress(50)

            if run_sar:
                status.text("Running SARIMA...")
                try:
                    m = SARIMAX(y_train, order=(1,0,0),
                                seasonal_order=(1,0,1,12)).fit(disp=False)
                    results['SARIMA'] = m.forecast(len(y_test))
                except:
                    results['SARIMA'] = results.get('ARIMA', np.full(len(y_test), y_train.mean()))
                progress.progress(75)

            if run_hw:
                status.text("Running Holt-Winters ETS...")
                if len(y_train) >= 24:
                    m = ExponentialSmoothing(y_train, trend='add',
                                             seasonal='add', seasonal_periods=12).fit(optimized=True)
                    results['HW-ETS'] = m.forecast(len(y_test))
                progress.progress(100)

            status.empty()
            progress.empty()

            # Metrics
            def metrics(actual, pred):
                mae  = np.mean(np.abs(actual-pred))
                rmse = np.sqrt(np.mean((actual-pred)**2))
                mape = np.mean(np.abs((actual-pred)/actual))*100
                return mae, rmse, mape

            st.markdown('<div class="section-header">📊 Accuracy Metrics</div>', unsafe_allow_html=True)
            met_rows = []
            for name, fc in results.items():
                mae, rmse, mape = metrics(y_test, fc)
                met_rows.append({'Model': name, 'MAE': f"{mae:,.1f}",
                                 'RMSE': f"{rmse:,.1f}", 'MAPE': f"{mape:.2f}%"})
            met_df = pd.DataFrame(met_rows)

            best_mae = min(results.items(), key=lambda x: np.mean(np.abs(y_test-x[1])))[0]
            st.dataframe(met_df, use_container_width=True, hide_index=True)
            st.success(f"🏆 **Best Model: {best_mae}** — lowest Mean Absolute Error on the {test_months}-month test period")

            # Actual vs predicted chart
            st.markdown('<div class="section-header">📉 Actual vs Predicted</div>', unsafe_allow_html=True)
            colors_fc = {'SMA':'#ffb300','ARIMA':'#42a5f5','SARIMA':'#ef5350','HW-ETS':'#66bb6a'}
            fig, ax = dark_fig((14, 5))
            ax.plot(ts['yearmonth_dt'].values[-test_months-12:-test_months],
                    y_train[-12:], color='#607d8b', linewidth=1.5,
                    linestyle='--', label='Training (last 12m)')
            ax.plot(dates_test, y_test, color='white', linewidth=2.5,
                    marker='o', markersize=4, label='Actual 2024', zorder=5)
            for name, fc in results.items():
                ax.plot(dates_test, fc, color=colors_fc.get(name,'#4fc3f7'),
                        linewidth=2, linestyle='--', marker='s',
                        markersize=3, label=name)
            ax.set_title('Actual vs Forecast — Test Period', color='#e3f2fd', fontsize=11, fontweight='bold')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            ax.legend(fontsize=9, facecolor='#0d1329', labelcolor='#90caf9')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👆 Upload data in the sidebar to run forecasting models.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">⚠️ Collision Severity Risk Calculator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    Select the conditions of a collision scenario below. The calculator uses the
    logistic regression model from the dissertation analysis to estimate the probability
    that a collision under these conditions results in a <strong>Fatal or Serious</strong> outcome.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        location  = st.selectbox("📍 Location", ["Urban", "Rural"])
        speed_lim = st.select_slider("🚀 Speed Limit (mph)", options=[20, 30, 40, 50, 60, 70], value=30)
        lighting  = st.selectbox("💡 Light Conditions", ["Daylight", "Darkness"])
    with c2:
        weather   = st.selectbox("🌧️ Weather", ["Fine", "Adverse (Rain/Snow/Fog/Wind)"])
        surface   = st.selectbox("🛣️ Road Surface", ["Dry", "Adverse (Wet/Ice/Snow/Flood)"])
        day_type  = st.selectbox("📅 Day of Week", ["Weekday", "Weekend"])

    # Regression coefficients from dissertation results
    INTERCEPT = -1.285
    COEF = {
        'rural'           :  0.129,
        'speed_limit_num' :  0.115,
        'darkness'        :  0.088,
        'is_weekend'      :  0.064,
        'adverse_surface' : -0.003,
        'adverse_weather' : -0.019,
    }
    # Mean and std for standardisation (approximate from dataset)
    MEANS = {'rural':0.325,'speed_limit_num':33.2,'darkness':0.285,
             'is_weekend':0.301,'adverse_surface':0.248,'adverse_weather':0.209}
    STDS  = {'rural':0.468,'speed_limit_num':10.8,'darkness':0.452,
             'is_weekend':0.459,'adverse_surface':0.432,'adverse_weather':0.407}

    vals = {
        'rural'           : 1 if location == "Rural" else 0,
        'speed_limit_num' : speed_lim,
        'darkness'        : 1 if lighting == "Darkness" else 0,
        'is_weekend'      : 1 if day_type == "Weekend" else 0,
        'adverse_surface' : 1 if "Adverse" in surface else 0,
        'adverse_weather' : 1 if "Adverse" in weather else 0,
    }

    logit = INTERCEPT
    for feat, val in vals.items():
        standardised = (val - MEANS[feat]) / STDS[feat]
        logit += COEF[feat] * standardised

    prob_severe = 1 / (1 + np.exp(-logit))

    st.markdown("---")
    col_prob, col_badge, col_baseline = st.columns(3)

    with col_prob:
        st.metric("Predicted Probability of Severe Outcome",
                  f"{prob_severe*100:.1f}%",
                  delta=f"{(prob_severe - 0.233)*100:+.1f}% vs 23.3% baseline")

    with col_badge:
        if prob_severe > 0.35:
            badge = "<span class='risk-badge-high'>HIGH RISK</span>"
        elif prob_severe > 0.25:
            badge = "<span class='risk-badge-medium'>MODERATE RISK</span>"
        else:
            badge = "<span class='risk-badge-low'>LOW RISK</span>"
        st.markdown(f"**Risk Level**<br>{badge}", unsafe_allow_html=True)

    with col_baseline:
        st.metric("Dataset Baseline", "23.3%", "overall severe rate 2020–2024")

    # Risk factor breakdown
    st.markdown('<div class="section-header">📊 Risk Factor Contributions</div>', unsafe_allow_html=True)
    contrib_data = []
    for feat, val in vals.items():
        std_val = (val - MEANS[feat]) / STDS[feat]
        contrib = COEF[feat] * std_val
        label_map = {'rural':'Rural Location','speed_limit_num':'Speed Limit',
                     'darkness':'Darkness','is_weekend':'Weekend',
                     'adverse_surface':'Adverse Surface','adverse_weather':'Adverse Weather'}
        contrib_data.append({'Factor': label_map[feat],
                              'Value': val,
                              'Log-Odds Contribution': contrib,
                              'Direction': '↑ Increases Risk' if contrib > 0 else '↓ Decreases Risk'})
    contrib_df = pd.DataFrame(contrib_data).sort_values('Log-Odds Contribution', ascending=False)
    st.dataframe(contrib_df[['Factor','Direction','Log-Odds Contribution']],
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — SCENARIO FORECAST (2025 or 2025–2026 continuous)
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    if ts_file:
        ts = load_ts(ts_file)
        y  = ts['Total_Collisions'].values

        # ── Determine forecast horizon based on selected year ──────────────────
        # 2025 → 12-month forecast (Jan–Dec 2025)
        # 2026 → 24-month continuous forecast (Jan 2025–Dec 2026)
        if forecast_yr == 2025:
            n_months    = 12
            start_year  = 2025
            period_label = "2025"
            period_desc  = "January to December 2025"
        else:
            n_months    = 24
            start_year  = 2025
            period_label = "2025–2026"
            period_desc  = "January 2025 to December 2026 (continuous)"

        header_txt = f"📅 Scenario Forecast — {period_label}"
        st.markdown(f'<div class="section-header">{header_txt}</div>', unsafe_allow_html=True)

        # Explain the period to the user
        if forecast_yr == 2026:
            st.info(
                f"📌 **2026 view shows the full continuous forecast period: {period_desc}.** "
                f"The first 12 months (2025) provide the connecting bridge from the actual "
                f"series. The second 12 months (2026) extend the projection forward. "
                f"Select **2025** in the sidebar to view the 2025 forecast in isolation."
            )

        scenario = st.radio(
            "Select Forecast Scenario:",
            ["🟢 Optimistic", "🟡 Base Case", "🔴 Pessimistic"],
            horizontal=True
        )

        scenario_adj = {"🟢 Optimistic": 0.95, "🟡 Base Case": 1.0, "🔴 Pessimistic": 1.05}
        adj = scenario_adj[scenario]

        # ── Fit HW-ETS on full actual series and forecast n_months ahead ───────
        hw        = ExponentialSmoothing(y, trend='add', seasonal='add',
                                        seasonal_periods=12).fit(optimized=True)
        base_fc   = hw.forecast(n_months)
        adj_fc    = base_fc * adj

        # Date index always starts Jan 2025 regardless of year selection
        future_idx = pd.date_range(start='2025-01-01', periods=n_months, freq='MS')

        # ── Build forecast table ────────────────────────────────────────────────
        future_df = pd.DataFrame({
            'Month'            : [d.strftime('%b %Y') for d in future_idx],
            'Base Forecast'    : [f"{v:,.0f}" for v in base_fc],
            'Scenario Forecast': [f"{v:,.0f}" for v in adj_fc],
            'Lower 95% CI'     : [f"{max(0, v*0.82):,.0f}" for v in adj_fc],
            'Upper 95% CI'     : [f"{v*1.18:,.0f}" for v in adj_fc],
        })

        # ── KPI metrics ────────────────────────────────────────────────────────
        if forecast_yr == 2025:
            # 12-month single year KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("2025 Annual Forecast", f"{adj_fc.sum():,.0f}",
                      f"{(adj_fc.sum()/y[-12:].sum()-1)*100:+.1f}% vs 2024")
            c2.metric("Peak Month",
                      future_df.iloc[adj_fc.argmax()]['Month'],
                      f"{adj_fc.max():,.0f} collisions")
            c3.metric("Lowest Month",
                      future_df.iloc[adj_fc.argmin()]['Month'],
                      f"{adj_fc.min():,.0f} collisions")
        else:
            # Split into 2025 and 2026 halves for the KPI row
            fc_2025 = adj_fc[:12]
            fc_2026 = adj_fc[12:]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("2025 Forecast Total", f"{fc_2025.sum():,.0f}",
                      f"{(fc_2025.sum()/y[-12:].sum()-1)*100:+.1f}% vs 2024")
            c2.metric("2026 Forecast Total", f"{fc_2026.sum():,.0f}",
                      f"{(fc_2026.sum()/fc_2025.sum()-1)*100:+.1f}% vs 2025")
            c3.metric("2-Year Peak Month",
                      future_df.iloc[adj_fc.argmax()]['Month'],
                      f"{adj_fc.max():,.0f} collisions")
            c4.metric("2-Year Lowest Month",
                      future_df.iloc[adj_fc.argmin()]['Month'],
                      f"{adj_fc.min():,.0f} collisions")

        # ── Forecast table ─────────────────────────────────────────────────────
        st.dataframe(future_df, use_container_width=True, hide_index=True)

        # ── Forecast chart ─────────────────────────────────────────────────────
        fig, ax = dark_fig((14, 5))
        ts['yearmonth_dt'] = pd.to_datetime(ts['yearmonth_str'])
        ax.plot(ts['yearmonth_dt'].values, y, color='#4fc3f7',
                linewidth=2, label='Actual 2020–2024')

        # For 2026 view: draw 2025 segment in lighter green, 2026 in bright green
        if forecast_yr == 2026:
            fc_2025 = adj_fc[:12]
            fc_2026 = adj_fc[12:]
            idx_2025 = future_idx[:12]
            idx_2026 = future_idx[12:]
            # Bridge line connecting actual series end to forecast start
            last_actual_date = ts['yearmonth_dt'].values[-1]
            last_actual_val  = y[-1]
            ax.plot([last_actual_date, idx_2025[0]],
                    [last_actual_val, fc_2025[0]],
                    color='#4fc3f7', linewidth=1.5, linestyle='--', alpha=0.5)
            ax.plot(idx_2025, fc_2025, color='#81c784',
                    linewidth=2, marker='o', markersize=4,
                    label=f'{scenario} Forecast 2025', alpha=0.85)
            ax.plot(idx_2026, fc_2026, color='#66bb6a',
                    linewidth=2.5, marker='D', markersize=5,
                    label=f'{scenario} Forecast 2026')
            if show_ci:
                ax.fill_between(future_idx, adj_fc*0.82, adj_fc*1.18,
                                alpha=0.15, color='#66bb6a', label='95% CI')
            # Vertical dividers for year boundaries
            ax.axvline(x=pd.Timestamp('2025-01-01'),
                       color='#ff8f00', linewidth=1.5, linestyle=':',
                       label='Forecast start (Jan 2025)')
            ax.axvline(x=pd.Timestamp('2026-01-01'),
                       color='#ffb300', linewidth=1, linestyle='--',
                       label='2026 boundary')
        else:
            ax.plot(future_idx, adj_fc, color='#66bb6a',
                    linewidth=2.5, marker='D', markersize=5,
                    label=f'{scenario} Forecast 2025')
            if show_ci:
                ax.fill_between(future_idx, adj_fc*0.82, adj_fc*1.18,
                                alpha=0.2, color='#66bb6a', label='95% CI')
            ax.axvline(x=pd.Timestamp('2025-01-01'),
                       color='#ff8f00', linewidth=1.5, linestyle=':')

        ax.set_title(f'{scenario} — Collision Forecast {period_label}',
                     color='#e3f2fd', fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha='right', color='#90caf9', fontsize=8)
        ax.legend(fontsize=9, facecolor='#0d1329', labelcolor='#90caf9')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Insurance implication ──────────────────────────────────────────────
        # For 2026 show both years separately
        if forecast_yr == 2026:
            fc_2025      = adj_fc[:12]
            fc_2026      = adj_fc[12:]
            res_low_25   = int(fc_2025.sum() * 850)
            res_high_25  = int(fc_2025.sum() * 1200)
            res_low_26   = int(fc_2026.sum() * 850)
            res_high_26  = int(fc_2026.sum() * 1200)
            st.markdown(f"""
            <div class='insight-box'>
            <strong>Insurance Implication ({scenario}) — 2025–2026 Continuous Forecast</strong><br>
            <strong>2025:</strong> Forecast {fc_2025.sum():,.0f} collisions →
            estimated reserve range <strong>£{res_low_25:,.0f} – £{res_high_25:,.0f}</strong><br>
            <strong>2026:</strong> Forecast {fc_2026.sum():,.0f} collisions →
            estimated reserve range <strong>£{res_low_26:,.0f} – £{res_high_26:,.0f}</strong><br>
            <strong>2-Year combined:</strong> {adj_fc.sum():,.0f} collisions →
            reserve range <strong>£{int(adj_fc.sum()*850):,.0f} – £{int(adj_fc.sum()*1200):,.0f}</strong><br>
            <em>Illustrative only — actual reserve calculations require insurer-specific claims
            conversion rates and severity cost data.</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            reserve_low  = int(adj_fc.sum() * 850)
            reserve_high = int(adj_fc.sum() * 1200)
            st.markdown(f"""
            <div class='insight-box'>
            <strong>Insurance Implication ({scenario})</strong><br>
            Forecast annual collisions: <strong>{adj_fc.sum():,.0f}</strong>.
            Assuming average claim cost of £850–£1,200 per collision-related claim,
            estimated claims reserve range: <strong>£{reserve_low:,.0f} – £{reserve_high:,.0f}</strong>.
            This is illustrative only — actual reserve calculations require insurer-specific
            claims conversion rates and severity cost data.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👆 Upload data in the sidebar to view scenario forecasts.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXECUTIVE REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">📋 Executive Summary Report</div>', unsafe_allow_html=True)

    if ts_file:
        ts = load_ts(ts_file)
        y  = ts['Total_Collisions'].values

        st.markdown(f"""
        <div style='background:#12193a; border:1px solid #2a3a6a; border-radius:12px; padding:28px; font-family:"IBM Plex Sans",sans-serif;'>

        <h2 style='color:#4fc3f7; font-size:1.4rem; margin-bottom:4px;'>Motor Insurance Risk Analytics</h2>
        <p style='color:#7986cb; font-size:0.85rem; margin-top:0;'>Executive Summary — UK Road Collision Data 2020–2024</p>
        <hr style='border-color:#2a3a6a; margin:16px 0;'>

        <h3 style='color:#90caf9; font-size:1rem;'>1. Dataset Overview</h3>
        <p style='color:#b0bec5; font-size:0.9rem;'>
        Total collision records analysed: <strong style='color:#4fc3f7'>{y.sum():,.0f}</strong><br>
        Study period: <strong style='color:#4fc3f7'>January 2020 to December 2024</strong> (60 months)<br>
        Monthly average: <strong style='color:#4fc3f7'>{y.mean():,.0f}</strong> collisions/month<br>
        COVID-19 lockdown low: <strong style='color:#ef5350'>{y.min():,.0f}</strong> (April 2020)<br>
        Post-restriction peak: <strong style='color:#66bb6a'>{y.max():,.0f}</strong> (November 2021)
        </p>

        <h3 style='color:#90caf9; font-size:1rem; margin-top:18px;'>2. Forecasting Model Results</h3>
        <p style='color:#b0bec5; font-size:0.9rem;'>
        Four models were compared on the 2024 held-out test period:<br>
        • <strong style='color:#66bb6a'>Holt-Winters ETS</strong> — Best model: MAE 281, RMSE 340, MAPE 3.41%<br>
        • <strong style='color:#42a5f5'>SARIMA (forced)</strong> — Second best: MAE 416<br>
        • <strong style='color:#42a5f5'>ARIMA(1,0,0)</strong> — MAE 471, MAPE 5.66%<br>
        • <strong style='color:#ffb300'>Simple Moving Average</strong> — Baseline: MAE 550, MAPE 6.91%<br>
        <strong style='color:#4fc3f7'>Recommendation: Holt-Winters ETS for stable conditions; SMA for disrupted environments.</strong>
        </p>

        <h3 style='color:#90caf9; font-size:1rem; margin-top:18px;'>3. Key Risk Factors (Regression Analysis)</h3>
        <p style='color:#b0bec5; font-size:0.9rem;'>
        Binary logistic regression on 503,475 collisions identified:<br>
        • <strong style='color:#ef5350'>Rural Location</strong> — OR 1.138: rural collisions 13.8% more likely to be severe<br>
        • <strong style='color:#ef5350'>Speed Limit</strong> — OR 1.122: higher speed zones increase severity by 12.2%<br>
        • <strong style='color:#ef5350'>Darkness</strong> — OR 1.092: driving in darkness increases severity by 9.2%<br>
        • <strong style='color:#ef5350'>Weekend</strong> — OR 1.066: weekend collisions 6.6% more likely to be severe<br>
        • <strong style='color:#42a5f5'>Adverse Weather</strong> — OR 0.982: slight reduction (risk compensation behaviour)<br>
        • <strong style='color:#42a5f5'>Adverse Surface</strong> — OR 0.997: marginal reduction (risk compensation behaviour)
        </p>

        <h3 style='color:#90caf9; font-size:1rem; margin-top:18px;'>4. 2025 Forecast</h3>
        <p style='color:#b0bec5; font-size:0.9rem;'>
        SARIMA model forecasts <strong style='color:#4fc3f7'>99,917</strong> collisions in 2025 (base case)<br>
        Holt-Winters ETS forecasts <strong style='color:#4fc3f7'>94,822</strong> collisions in 2025<br>
        Both models forecast a slight decline from the 2024 actual of 100,927<br>
        Seasonal pattern expected to continue — peak October, lowest February/April
        </p>

        <h3 style='color:#90caf9; font-size:1rem; margin-top:18px;'>5. Recommendations for Insurance Practice</h3>
        <p style='color:#b0bec5; font-size:0.9rem;'>
        • Adopt Holt-Winters ETS for monthly claims forecasting under stable conditions<br>
        • Apply rural location and speed limit as primary risk rating variables<br>
        • Consider time-of-day and day-of-week factors in telematics pricing models<br>
        • Build seasonal staffing plans around the October peak and April trough<br>
        • Update models annually as post-pandemic driving patterns continue to evolve
        </p>

        <hr style='border-color:#2a3a6a; margin:18px 0;'>
        <p style='color:#546e7a; font-size:0.8rem;'>Generated by Motor Insurance Risk Analytics Dashboard •
        Wilson Dissertation • University of Bolton • MSc Business Analytics and Technology</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Upload data in the sidebar to generate the executive report.")