import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import erfinv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Kala Ulang Banjir",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .distribution-card {
        border-left: 4px solid;
        padding: 0.5rem;
        margin: 0.5rem 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi utilitas
def ensure_numeric_array(x, name='data'):
    """Convert input to 1D numpy array of floats. Accept dict, list, pandas Series/ndarray."""
    if isinstance(x, dict):
        x = list(x.values())
    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(x, dtype=float)
    else:
        try:
            arr = np.asarray(list(x), dtype=float)
        except Exception:
            raise ValueError(f"{name} must be array-like numeric. Got {type(x)}")
    if arr.ndim != 1:
        arr = arr.flatten()
    return arr

def calculate_statistics(data):
    """Hitung statistik dasar data"""
    data = ensure_numeric_array(data, 'data')
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'cv': float(np.std(data) / np.mean(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'max': float(np.max(data)),
        'min': float(np.min(data)),
        'median': float(np.median(data)),
        'count': int(len(data))
    }

def fit_gumbel(data):
    """Fitting distribusi Gumbel"""
    data = ensure_numeric_array(data, 'data')
    mu = np.mean(data)
    sigma = np.std(data)
    beta = sigma * np.sqrt(6) / np.pi
    mu0 = mu - 0.5772 * beta
    return {'mu0': mu0, 'beta': beta}

def fit_lognormal(data):
    """Fitting distribusi Log-Normal"""
    data = ensure_numeric_array(data, 'data')
    if np.any(data <= 0):
        raise ValueError('All data must be positive for LogNormal fitting.')
    y = np.log(data)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    return {'mu': mu_y, 'sigma': sigma_y}

def fit_logpearson3(data):
    """Fitting distribusi Log-Pearson III"""
    data = ensure_numeric_array(data, 'data')
    if np.any(data <= 0):
        raise ValueError('All data must be positive for LogPearson3 fitting.')
    y = np.log(data)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    skew = float(stats.skew(y))
    return {'mu': mu_y, 'sigma': sigma_y, 'skew': skew}

def calculate_frequency_factor(skew, P):
    """Hitung faktor frekuensi K untuk Log-Pearson III"""
    if abs(skew) < 1e-10:
        return stats.norm.ppf(P)
    else:
        Z = stats.norm.ppf(P)
        K = (2/skew) * ((1 + (skew*Z)/6 - (skew**2)/36)**3 - 1)
        return K

def calculate_return_periods(dist_type, params, return_periods):
    """Hitung debit untuk berbagai kala ulang"""
    discharges = {}
    
    for T in return_periods:
        P = 1 - 1/T
        
        if dist_type == 'Gumbel':
            mu0, beta = params['mu0'], params['beta']
            Kt = -np.sqrt(6)/np.pi * (0.5772 + np.log(np.log(1/P)))
            Q = mu0 + Kt * beta
            
        elif dist_type == 'Normal':
            mu, sigma = params['mu'], params['sigma']
            Z = stats.norm.ppf(P)
            Q = mu + Z * sigma
            
        elif dist_type == 'LogNormal':
            mu, sigma = params['mu'], params['sigma']
            Z = stats.norm.ppf(P)
            Q = np.exp(mu + Z * sigma)
            
        elif dist_type == 'LogPearson3':
            mu, sigma, skew = params['mu'], params['sigma'], params['skew']
            K = calculate_frequency_factor(skew, P)
            Q = np.exp(mu + K * sigma)
        
        discharges[T] = float(Q)
    
    return discharges

def chi_square_test(data, dist_type, params, alpha=0.05):
    """Uji kesesuaian Chi-Square"""
    data = ensure_numeric_array(data, 'data')
    n = len(data)
    k = max(3, int(1 + 3.322 * np.log10(n)))  # Aturan Sturges
    
    # Transformasi data untuk distribusi log
    if dist_type in ['LogNormal', 'LogPearson3']:
        if np.any(data <= 0):
            raise ValueError('All data must be positive for log-transform in Chi-Square test.')
        data_fit = np.log(data)
    else:
        data_fit = data
    
    # Buat bins
    min_val, max_val = np.min(data_fit), np.max(data_fit)
    bins = np.linspace(min_val, max_val, k+1)
    
    # Frekuensi observasi
    observed, _ = np.histogram(data_fit, bins=bins)
    
    # Fungsi CDF berdasarkan distribusi
    if dist_type == 'Gumbel':
        cdf_func = lambda x: stats.gumbel_r.cdf(x, loc=params['mu0'], scale=params['beta'])
    elif dist_type == 'Normal':
        cdf_func = lambda x: stats.norm.cdf(x, loc=params['mu'], scale=params['sigma'])
    elif dist_type == 'LogNormal':
        cdf_func = lambda x: stats.norm.cdf(x, loc=params['mu'], scale=params['sigma'])
    elif dist_type == 'LogPearson3':
        if abs(params['skew']) < 1e-10:
            cdf_func = lambda x: stats.norm.cdf(x, loc=params['mu'], scale=params['sigma'])
        else:
            alpha_param = 4/(params['skew']**2)
            beta_param = params['sigma'] * abs(params['skew']) / 2
            loc_param = params['mu'] - 2*params['sigma']/params['skew']
            cdf_func = lambda x: stats.gamma.cdf(x, a=alpha_param, loc=loc_param, scale=beta_param)
    
    # Frekuensi harapan
    expected = []
    for i in range(k):
        prob = cdf_func(bins[i+1]) - cdf_func(bins[i])
        expected.append(prob * n)
    
    expected = np.array(expected)
    # Hindari pembagian dengan nol
    valid_idx = expected > 0
    if np.sum(valid_idx) < 2:
        return {'statistic': np.nan, 'p_value': 0, 'passed': False, 'df': 0}
    
    chi2_stat = np.sum((observed[valid_idx] - expected[valid_idx])**2 / expected[valid_idx])
    df = np.sum(valid_idx) - len(params) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, max(df, 1))
    
    return {
        'statistic': float(chi2_stat),
        'p_value': float(p_value),
        'passed': p_value > alpha,
        'df': int(df)
    }

def kolmogorov_smirnov_test(data, dist_type, params, alpha=0.05):
    """Uji kesesuaian Kolmogorov-Smirnov"""
    data = ensure_numeric_array(data, 'data')
    n = len(data)
    
    # Transformasi dan urutkan data
    if dist_type in ['LogNormal', 'LogPearson3']:
        if np.any(data <= 0):
            raise ValueError('All data must be positive for log-transform in KS test.')
        sorted_data = np.sort(np.log(data))
    else:
        sorted_data = np.sort(data)
    
    # CDF empiris
    ecdf = np.arange(1, n+1) / n
    
    # CDF teoritis
    if dist_type == 'Gumbel':
        tcdf = stats.gumbel_r.cdf(sorted_data, loc=params['mu0'], scale=params['beta'])
    elif dist_type == 'Normal':
        tcdf = stats.norm.cdf(sorted_data, loc=params['mu'], scale=params['sigma'])
    elif dist_type == 'LogNormal':
        tcdf = stats.norm.cdf(sorted_data, loc=params['mu'], scale=params['sigma'])
    elif dist_type == 'LogPearson3':
        if abs(params['skew']) < 1e-10:
            tcdf = stats.norm.cdf(sorted_data, loc=params['mu'], scale=params['sigma'])
        else:
            alpha_param = 4/(params['skew']**2)
            beta_param = params['sigma'] * abs(params['skew']) / 2
            loc_param = params['mu'] - 2*params['sigma']/params['skew']
            tcdf = stats.gamma.cdf(sorted_data, a=alpha_param, loc=loc_param, scale=beta_param)
    
    # Statistik KS
    D = np.max(np.abs(ecdf - tcdf))
    D_critical = 1.36 / np.sqrt(n)  # untuk alpha=0.05
    
    return {
        'statistic': float(D),
        'critical': float(D_critical),
        'passed': D < D_critical
    }

def select_best_distribution(test_results):
    """Pilih distribusi terbaik berdasarkan hasil uji"""
    # Kriteria: Lolos kedua uji, kemudian statistik KS terkecil
    passed_distributions = []
    
    for dist_name, tests in test_results.items():
        chi2_passed = tests['chi2']['passed']
        ks_passed = tests['ks']['passed']
        
        if chi2_passed and ks_passed:
            passed_distributions.append((dist_name, tests['ks']['statistic']))
    
    if passed_distributions:
        passed_distributions.sort(key=lambda x: x[1])
        return passed_distributions[0][0]
    else:
        # Jika tidak ada yang lolos, pilih dengan KS terkecil
        all_dists = [(dist, tests['ks']['statistic']) for dist, tests in test_results.items()]
        all_dists.sort(key=lambda x: x[1])
        return all_dists[0][0]

def create_probability_plot(data, distributions, best_dist):
    """Buat plot probabilitas menggunakan Plotly"""
    # Urutkan data
    sorted_data = np.sort(data)[::-1]
    n = len(data)
    
    # Hitung probabilitas empiris (Weibull)
    ranks = np.arange(1, n+1)
    P = ranks / (n + 1)
    T_empirical = 1 / (1 - P)
    
    # Warna untuk setiap distribusi
    colors = {
        'Gumbel': '#FF6B6B',
        'Normal': '#4ECDC4',
        'LogNormal': '#45B7D1',
        'LogPearson3': '#96CEB4',
        'Empirical': '#2C3E50'
    }
    
    # Buat subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Plot Probabilitas', 'Plot Kala Ulang',
                       'Distribusi Teoritis vs Data', 'QQ-Plot'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Plot Probabilitas (kiri atas)
    fig.add_trace(
        go.Scatter(
            x=-np.log(-np.log(P)),
            y=sorted_data,
            mode='markers',
            name='Data Empiris',
            marker=dict(color=colors['Empirical'], size=8),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Tambahkan garis teoritis untuk setiap distribusi
    P_theo = np.linspace(0.01, 0.99, 100)
    x_plot = -np.log(-np.log(P_theo))
    
    for dist_name, params in distributions.items():
        if params is None:
            continue
            
        if dist_name == 'Gumbel':
            mu0, beta = params['mu0'], params['beta']
            Q_theo = mu0 - beta * np.log(-np.log(P_theo))
            
        elif dist_name == 'Normal':
            mu, sigma = params['mu'], params['sigma']
            Q_theo = mu + sigma * stats.norm.ppf(P_theo)
            
        elif dist_name == 'LogNormal':
            mu, sigma = params['mu'], params['sigma']
            Q_theo = np.exp(mu + sigma * stats.norm.ppf(P_theo))
            
        elif dist_name == 'LogPearson3':
            mu, sigma, skew = params['mu'], params['sigma'], params['skew']
            Q_theo = []
            for p in P_theo:
                K = calculate_frequency_factor(skew, p)
                Q_theo.append(np.exp(mu + K * sigma))
            Q_theo = np.array(Q_theo)
        
        line_width = 3 if dist_name == best_dist else 1.5
        line_dash = 'solid' if dist_name == best_dist else 'dash'
        
        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=Q_theo,
                mode='lines',
                name=dist_name,
                line=dict(color=colors[dist_name], width=line_width, dash=line_dash),
                showlegend=True
            ),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="Variabel Tereduksi [-ln(-ln(P))]", row=1, col=1)
    fig.update_yaxes(title_text="Debit (m³/detik)", row=1, col=1)
    
    # 2. Plot Kala Ulang (kanan atas)
    fig.add_trace(
        go.Scatter(
            x=T_empirical,
            y=sorted_data,
            mode='markers',
            name='Data Empiris',
            marker=dict(color=colors['Empirical'], size=8),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Tambahkan garis teoritis untuk kala ulang
    T_theo = np.logspace(0, 3, 100)  # 1 sampai 1000 tahun
    
    for dist_name, params in distributions.items():
        if params is None:
            continue
            
        discharges = calculate_return_periods(dist_name, params, T_theo)
        Q_theo = list(discharges.values())
        
        line_width = 3 if dist_name == best_dist else 1.5
        line_dash = 'solid' if dist_name == best_dist else 'dash'
        
        fig.add_trace(
            go.Scatter(
                x=T_theo,
                y=Q_theo,
                mode='lines',
                name=dist_name,
                line=dict(color=colors[dist_name], width=line_width, dash=line_dash),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Kala Ulang (tahun)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Debit (m³/detik)", row=1, col=2)
    
    # 3. Distribusi Teoritis vs Data (kiri bawah)
    # Histogram data
    fig.add_trace(
        go.Histogram(
            x=data,
            name='Data',
            histnorm='probability density',
            marker_color=colors['Empirical'],
            opacity=0.5,
            nbinsx=20,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # PDF teoritis
    x_pdf = np.linspace(np.min(data)*0.5, np.max(data)*1.5, 1000)
    
    for dist_name, params in distributions.items():
        if params is None:
            continue
            
        if dist_name == 'Gumbel':
            pdf = stats.gumbel_r.pdf(x_pdf, loc=params['mu0'], scale=params['beta'])
        elif dist_name == 'Normal':
            pdf = stats.norm.pdf(x_pdf, loc=params['mu'], scale=params['sigma'])
        elif dist_name == 'LogNormal':
            pdf = stats.lognorm.pdf(x_pdf, s=params['sigma'], scale=np.exp(params['mu']))
        elif dist_name == 'LogPearson3':
            # Lewati untuk sementara
            continue
        
        line_width = 2 if dist_name == best_dist else 1
        line_dash = 'solid' if dist_name == best_dist else 'dash'
        
        fig.add_trace(
            go.Scatter(
                x=x_pdf,
                y=pdf,
                mode='lines',
                name=f"{dist_name} (PDF)",
                line=dict(color=colors[dist_name], width=line_width, dash=line_dash),
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Debit (m³/detik)", row=2, col=1)
    fig.update_yaxes(title_text="Densitas Probabilitas", row=2, col=1)
    
    # 4. QQ-Plot (kanan bawah)
    # Hitung quantiles teoritis
    theoretical_quantiles = {}
    P_qq = np.arange(1, n+1) / (n+1)
    
    for dist_name, params in distributions.items():
        if params is None:
            continue
            
        if dist_name == 'Gumbel':
            Q_theo = params['mu0'] - params['beta'] * np.log(-np.log(P_qq))
        elif dist_name == 'Normal':
            Q_theo = params['mu'] + params['sigma'] * stats.norm.ppf(P_qq)
        elif dist_name == 'LogNormal':
            Q_theo = np.exp(params['mu'] + params['sigma'] * stats.norm.ppf(P_qq))
        elif dist_name == 'LogPearson3':
            Q_theo = []
            for p in P_qq:
                K = calculate_frequency_factor(params['skew'], p)
                Q_theo.append(np.exp(params['mu'] + K * params['sigma']))
            Q_theo = np.array(Q_theo)
        
        theoretical_quantiles[dist_name] = Q_theo
        
        marker_size = 8 if dist_name == best_dist else 6
        
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=Q_theo,
                mode='markers',
                name=dist_name,
                marker=dict(color=colors[dist_name], size=marker_size),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Garis diagonal sempurna
    min_val = min(np.min(sorted_data), np.min(list(theoretical_quantiles.values())[0]))
    max_val = max(np.max(sorted_data), np.max(list(theoretical_quantiles.values())[0]))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Garis Sempurna',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Data Empiris", row=2, col=2)
    fig.update_yaxes(title_text="Quantiles Teoritis", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"Analisis Frekuensi Banjir - Distribusi Terbaik: {best_dist}",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig

def create_comparison_table(results, return_periods):
    """Buat tabel perbandingan debit untuk semua distribusi"""
    data = []
    for T in return_periods:
        row = {'Kala Ulang (tahun)': T}
        for dist_name, dist_results in results['results'].items():
            row[dist_name] = dist_results['discharges'][T]
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def create_download_link(df, filename="hasil_analisis.csv"):
    """Buat link download untuk dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Header aplikasi
st.markdown('<h1 class="main-header">🌊 Analisis Kala Ulang Banjir</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p>Aplikasi untuk analisis frekuensi banjir dengan pemilihan distribusi terbaik<br>
    (Gumbel, Normal, Log-Normal, Log-Pearson III) menggunakan uji kesesuaian statistik</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk input
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Analisis")
    
    # Input metode data
    data_source = st.radio(
        "Sumber Data:",
        ["Upload File", "Input Manual", "Gunakan Data Contoh"]
    )
    
    # Input kala ulang
    st.markdown("### 📅 Kala Ulang")
    return_periods_input = st.text_input(
        "Kala ulang (pisahkan dengan koma):",
        value="2, 5, 10, 25, 50, 100"
    )
    
    # Parse kala ulang
    try:
        return_periods = [float(x.strip()) for x in return_periods_input.split(',')]
        return_periods = [x for x in return_periods if x > 0]
    except:
        return_periods = [2, 5, 10, 25, 50, 100]
    
    # Tingkat signifikansi
    alpha = st.slider(
        "Tingkat Signifikansi (α):",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Digunakan untuk uji kesesuaian distribusi"
    )
    
    # Tombol analisis
    analyze_button = st.button("🚀 Mulai Analisis", type="primary", width='stretch')
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.info("""
    **Distribusi yang dianalisis:**
    1. **Gumbel** - Cocok untuk data ekstrim
    2. **Normal** - Untuk data simetris
    3. **Log-Normal** - Untuk data positif miring
    4. **Log-Pearson III** - Fleksibel untuk berbagai skewness
    """)

# Session state untuk menyimpan data
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Konten utama - Input Data
st.markdown('<h2 class="sub-header">📥 Input Data</h2>', unsafe_allow_html=True)

# Kontainer untuk input data
input_container = st.container()

with input_container:
    col1, col2 = st.columns(2)
    
    with col1:
        if data_source == "Upload File":
            st.markdown("#### Upload File Data")
            uploaded_file = st.file_uploader(
                "Pilih file CSV, TXT, atau Excel",
                type=['csv', 'txt', 'xlsx', 'xls']
            )
            
            if uploaded_file is not None:
                try:
                    # Baca file berdasarkan tipe
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.txt'):
                        df = pd.read_csv(uploaded_file, delimiter='\t')
                    else:  # Excel
                        df = pd.read_excel(uploaded_file)
                    
                    # Tampilkan preview
                    st.write("**Preview Data:**")
                    st.dataframe(df.head(), width='stretch')
                    
                    # Pilih kolom
                    if len(df.columns) > 1:
                        column = st.selectbox("Pilih kolom data debit:", df.columns)
                    else:
                        column = df.columns[0]
                    
                    # Ekstrak data
                    data = df[column].dropna().astype(float).values
                    
                    if len(data) >= 10:
                        st.session_state.flood_data = data
                        st.success(f"✅ Data berhasil dimuat: {len(data)} titik data")
                        
                        # Tampilkan statistik
                        stats = calculate_statistics(data)
                        st.markdown("**Statistik Data:**")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Jumlah Data", stats['count'])
                        with col_stat2:
                            st.metric("Rata-rata", f"{stats['mean']:.2f}")
                        with col_stat3:
                            st.metric("Std Dev", f"{stats['std']:.2f}")
                        with col_stat4:
                            st.metric("Skewness", f"{stats['skewness']:.3f}")
                    else:
                        st.warning("⚠️ Data terlalu sedikit. Minimal 10 titik data diperlukan.")
                        
                except Exception as e:
                    st.error(f"Error membaca file: {str(e)}")
        
        elif data_source == "Input Manual":
            st.markdown("#### Input Data Manual")
            manual_data = st.text_area(
                "Masukkan data debit (pisahkan dengan spasi, koma, atau baris baru):",
                height=150,
                value="125.3 98.7 156.2 210.5 178.9 134.2 189.5 167.8 145.6 201.3 165.4 192.7 178.3 154.9 203.1 167.5 189.2 176.8 198.4 163.9"
            )
            
            if manual_data:
                try:
                    # Parse data
                    data_list = []
                    for value in manual_data.replace(',', ' ').replace('\n', ' ').split():
                        try:
                            data_list.append(float(value))
                        except ValueError:
                            continue
                    
                    if len(data_list) >= 10:
                        data = np.array(data_list)
                        st.session_state.flood_data = data
                        st.success(f"✅ Data berhasil dimasukkan: {len(data)} titik data")
                        
                        # Tampilkan statistik
                        stats = calculate_statistics(data)
                        st.markdown("**Statistik Data:**")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Jumlah Data", stats['count'])
                        with col_stat2:
                            st.metric("Rata-rata", f"{stats['mean']:.2f}")
                        with col_stat3:
                            st.metric("Std Dev", f"{stats['std']:.2f}")
                        with col_stat4:
                            st.metric("Skewness", f"{stats['skewness']:.3f}")
                    else:
                        st.warning("⚠️ Data terlalu sedikit. Minimal 10 titik data diperlukan.")
                        
                except Exception as e:
                    st.error(f"Error memproses data: {str(e)}")
        
        else:  # Data Contoh
            st.markdown("#### Data Contoh")
            st.info("Data debit banjir contoh untuk 30 tahun (dalam m³/detik)")
            
            # Generate contoh data
            np.random.seed(42)
            n_example = 30
            log_mu = 5.0
            log_sigma = 0.4
            example_data = np.exp(np.random.normal(log_mu, log_sigma, n_example))
            
            # Tampilkan data
            example_df = pd.DataFrame({
                'Tahun': range(1, n_example + 1),
                'Debit (m³/detik)': example_data.round(1)
            })
            st.dataframe(example_df, width='stretch')
            
            # Tombol untuk menggunakan data contoh
            if st.button("Gunakan Data Contoh", width='stretch'):
                st.session_state.flood_data = example_data
                st.success(f"✅ Data contoh berhasil dimuat: {len(example_data)} titik data")
                
                # Tampilkan statistik
                stats = calculate_statistics(example_data)
                st.markdown("**Statistik Data:**")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Jumlah Data", stats['count'])
                with col_stat2:
                    st.metric("Rata-rata", f"{stats['mean']:.2f}")
                with col_stat3:
                    st.metric("Std Dev", f"{stats['std']:.2f}")
                with col_stat4:
                    st.metric("Skewness", f"{stats['skewness']:.3f}")
    
    with col2:
        st.markdown("#### 📊 Visualisasi Data")
        if st.session_state.flood_data is not None:
            data = st.session_state.flood_data
            
            # Plot histogram
            fig_hist = px.histogram(
                x=data,
                nbins=20,
                title="Distribusi Data Debit",
                labels={'x': 'Debit (m³/detik)', 'y': 'Frekuensi'}
            )
            fig_hist.update_traces(marker_color='#3B82F6', opacity=0.7)
            st.plotly_chart(fig_hist, width='stretch')
            
            # Box plot
            fig_box = px.box(
                y=data,
                title="Box Plot Data Debit",
                labels={'y': 'Debit (m³/detik)'}
            )
            fig_box.update_traces(marker_color='#10B981')
            st.plotly_chart(fig_box, width='stretch')
        else:
            st.info("Upload atau input data untuk melihat visualisasi")

# Analisis Data
if analyze_button and st.session_state.flood_data is not None:
    st.markdown('<h2 class="sub-header">📈 Hasil Analisis</h2>', unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data = st.session_state.flood_data
    n = len(data)
   
    # Step 1: Fitting distribusi
    status_text.text("Fitting distribusi...")
    progress_bar.progress(25)
    
    distributions = {
        'Gumbel': fit_gumbel(data),
        'Normal': {'mu': np.mean(data), 'sigma': np.std(data)},
        'LogNormal': fit_lognormal(data),
        'LogPearson3': fit_logpearson3(data)
    }
    
    # Step 2: Hitung debit untuk kala ulang
    status_text.text("Menghitung debit untuk kala ulang...")
    progress_bar.progress(50)
    
    results = {}
    for dist_name, params in distributions.items():
        discharges = calculate_return_periods(dist_name, params, return_periods)
        results[dist_name] = {
            'parameters': params,
            'discharges': discharges
        }
    
    # Step 3: Uji kesesuaian
    status_text.text("Melakukan uji kesesuaian...")
    progress_bar.progress(75)
    
    test_results = {}
    for dist_name, params in distributions.items():
        chi2_result = chi_square_test(data, dist_name, params, alpha)
        ks_result = kolmogorov_smirnov_test(data, dist_name, params, alpha)
        
        test_results[dist_name] = {
            'chi2': chi2_result,
            'ks': ks_result
        }
    
    # Step 4: Pilih distribusi terbaik
    status_text.text("Memilih distribusi terbaik...")
    progress_bar.progress(90)
    
    best_distribution = select_best_distribution(test_results)
    
    # Step 5: Buat plot
    status_text.text("Membuat visualisasi...")
    plot_fig = create_probability_plot(data, distributions, best_distribution)
    
    progress_bar.progress(100)
    status_text.text("Analisis selesai!")
    
    # Simpan hasil ke session state
    st.session_state.analysis_results = {
        'best_distribution': best_distribution,
        'return_periods': return_periods,
        'results': results,
        'tests': test_results,
        'data_statistics': calculate_statistics(data),
        'plot_fig': plot_fig
    }
    
    # Beri jeda kecil sebelum menampilkan hasil
    import time
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

# Tampilkan hasil jika tersedia
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    
    # Tampilkan distribusi terbaik
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    col_best1, col_best2 = st.columns([3, 1])
    with col_best1:
        st.markdown(f"### 🏆 Distribusi Terbaik: **{results['best_distribution']}**")
        st.markdown(f"Distribusi ini dipilih berdasarkan uji kesesuaian Chi-Square dan Kolmogorov-Smirnov")
    with col_best2:
        st.metric("Jumlah Data", results['data_statistics']['count'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab untuk hasil
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visualisasi", 
        "📋 Hasil Uji", 
        "💧 Debit Kala Ulang", 
        "📥 Download"
    ])
    
    with tab1:
        # Tampilkan plot
        st.plotly_chart(results['plot_fig'], width='stretch')
        
        # Tambahan visualisasi
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Plot perbandingan debit untuk distribusi terbaik
            best_dist = results['best_distribution']
            best_discharges = results['results'][best_dist]['discharges']
            
            fig_best = go.Figure()
            fig_best.add_trace(go.Scatter(
                x=list(best_discharges.keys()),
                y=list(best_discharges.values()),
                mode='lines+markers',
                name=best_dist,
                line=dict(color='#FF6B6B', width=3)
            ))
            
            fig_best.update_layout(
                title=f"Debit Banjir - {best_dist}",
                xaxis_title="Kala Ulang (tahun)",
                yaxis_title="Debit (m³/detik)",
                xaxis_type="log",
                height=400
            )
            st.plotly_chart(fig_best, width='stretch')
        
        with col_viz2:
            # Plot perbandingan semua distribusi untuk Q100
            q100_values = {}
            for dist_name, dist_results in results['results'].items():
                q100_values[dist_name] = dist_results['discharges'][100]
            
            fig_compare = go.Figure(data=[
                go.Bar(
                    x=list(q100_values.keys()),
                    y=list(q100_values.values()),
                    marker_color=['#FF6B6B' if d == best_dist else '#4ECDC4' for d in q100_values.keys()]
                )
            ])
            
            fig_compare.update_layout(
                title="Perbandingan Debit Q100",
                xaxis_title="Distribusi",
                yaxis_title="Debit (m³/detik)",
                height=400
            )
            st.plotly_chart(fig_compare, width='stretch')
    
    with tab2:
        st.markdown("### 📊 Hasil Uji Kesesuaian Distribusi")
        
        # Tampilkan hasil uji dalam tabel
        test_data = []
        for dist_name, tests in results['tests'].items():
            chi2_passed = "✅" if tests['chi2']['passed'] else "❌"
            ks_passed = "✅" if tests['ks']['passed'] else "❌"
            
            test_data.append({
                'Distribusi': dist_name,
                'Chi-Square (χ²)': f"{tests['chi2']['statistic']:.4f}",
                'p-value': f"{tests['chi2']['p_value']:.4f}",
                'Lolos Chi-Square': chi2_passed,
                'KS (D)': f"{tests['ks']['statistic']:.4f}",
                'D kritis': f"{tests['ks']['critical']:.4f}",
                'Lolos KS': ks_passed,
                'Status': '✅ Direkomendasikan' if tests['chi2']['passed'] and tests['ks']['passed'] else '⚠️ Tidak Direkomendasikan'
            })
        
        test_df = pd.DataFrame(test_data)
        st.markdown("#### Ringkasan Hasil Uji")
        st.dataframe(
            test_df.style.apply(
                lambda x: ['background-color: #D1FAE5' if 'Direkomendasikan' in str(v) else '' for v in x], 
                axis=1
            ),
            width='stretch'
        )

        # Tabel rinci untuk masing-masing metode uji
        chi2_data = []
        ks_data = []
        for dist_name, tests in results['tests'].items():
            chi2 = tests['chi2']
            chi2_data.append({
                'Distribusi': dist_name,
                'Chi2_Statistic': chi2.get('statistic', np.nan),
                'Chi2_pValue': chi2.get('p_value', np.nan),
                'Chi2_Passed': chi2.get('passed', False),
                'df': chi2.get('df', None)
            })
            ks = tests['ks']
            ks_data.append({
                'Distribusi': dist_name,
                'KS_Statistic': ks.get('statistic', np.nan),
                'KS_Critical': ks.get('critical', np.nan),
                'KS_Passed': ks.get('passed', False)
            })

        chi2_df = pd.DataFrame(chi2_data)
        ks_df = pd.DataFrame(ks_data)

        st.markdown("#### Hasil Uji per Metode")
        col_chi, col_ks = st.columns(2)
        with col_chi:
            st.markdown("**Chi-Square**")
            st.dataframe(chi2_df, width='stretch')
        with col_ks:
            st.markdown("**Kolmogorov-Smirnov**")
            st.dataframe(ks_df, width='stretch')

        # Penjelasan uji
        with st.expander("📝 Interpretasi Hasil Uji"):
            st.markdown("""
            **Chi-Square Test:**
            - **Hipotesis Null**: Data mengikuti distribusi tertentu
            - **Kriteria**: p-value > α (0.05) → Terima H0 (distribusi sesuai)
            - **Kekurangan**: Sensitif terhadap banyaknya kelas interval
            
            **Kolmogorov-Smirnov Test:**
            - **Hipotesis Null**: Data mengikuti distribusi tertentu
            - **Kriteria**: D < D_kritis → Terima H0 (distribusi sesuai)
            - **Kelebihan**: Tidak tergantung pada banyaknya kelas interval
            
            **Distribusi direkomendasikan jika lolos kedua uji.**
            """)
    
    with tab3:
        st.markdown("### 💧 Debit Banjir untuk Berbagai Kala Ulang")
        
        # Tabel perbandingan semua distribusi
        comparison_df = create_comparison_table(results, results['return_periods'])
        
        # Format tabel
        styled_df = comparison_df.style.format({
            'Gumbel': '{:.2f}',
            'Normal': '{:.2f}',
            'LogNormal': '{:.2f}',
            'LogPearson3': '{:.2f}'
        }).apply(
            # Apply bold style to all cells in the best distribution column
            lambda col: ['font-weight: bold' if col.name == results['best_distribution'] else '' for _ in col],
            axis=0
        )
        
        st.dataframe(styled_df, width='stretch')

        # Tabel detail nilai untuk kala ulang
        st.markdown("#### Tabel Debit Kala Ulang (Detail Nilai)")
        st.dataframe(comparison_df, width='stretch')
        
        # Tampilkan parameter distribusi
        st.markdown("#### 📐 Parameter Distribusi")
        
        param_cols = st.columns(4)
        for idx, (dist_name, dist_results) in enumerate(results['results'].items()):
            with param_cols[idx]:
                st.markdown(f"**{dist_name}**")
                params = dist_results['parameters']
                if dist_name == 'Gumbel':
                    st.write(f"μ₀ = {params['mu0']:.4f}")
                    st.write(f"β = {params['beta']:.4f}")
                elif dist_name == 'Normal':
                    st.write(f"μ = {params['mu']:.4f}")
                    st.write(f"σ = {params['sigma']:.4f}")
                elif dist_name == 'LogNormal':
                    st.write(f"μ = {params['mu']:.4f}")
                    st.write(f"σ = {params['sigma']:.4f}")
                elif dist_name == 'LogPearson3':
                    st.write(f"μ = {params['mu']:.4f}")
                    st.write(f"σ = {params['sigma']:.4f}")
                    st.write(f"skew = {params['skew']:.4f}")
    
    with tab4:
        st.markdown("### 📥 Download Hasil Analisis")
        
        # Buat dataframe lengkap untuk download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data statistik
        stats_data = []
        for key, value in results['data_statistics'].items():
            stats_data.append({'Parameter': key, 'Nilai': value})
        stats_df = pd.DataFrame(stats_data)
        
        # Data debit kala ulang
        discharge_df = comparison_df.copy()
        
        # Data uji kesesuaian
        test_download_data = []
        for dist_name, tests in results['tests'].items():
            test_download_data.append({
                'Distribusi': dist_name,
                'Chi2_Statistic': tests['chi2']['statistic'],
                'Chi2_pValue': tests['chi2']['p_value'],
                'Chi2_Passed': tests['chi2']['passed'],
                'KS_Statistic': tests['ks']['statistic'],
                'KS_Critical': tests['ks']['critical'],
                'KS_Passed': tests['ks']['passed']
            })
        test_download_df = pd.DataFrame(test_download_data)
        
        # Gabungkan semua data ke dalam satu Excel file
        output = io.BytesIO()
        # Use openpyxl engine which is already in requirements to avoid external dependency on xlsxwriter
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Ringkasan
            summary_data = {
                'Item': ['Distribusi Terbaik', 'Tanggal Analisis', 'Jumlah Data', 'Tingkat Signifikansi'],
                'Nilai': [results['best_distribution'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         results['data_statistics']['count'], alpha]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Ringkasan', index=False)
            
            # Sheet 2: Statistik Data
            stats_df.to_excel(writer, sheet_name='Statistik Data', index=False)
            
            # Sheet 3: Debit Kala Ulang
            discharge_df.to_excel(writer, sheet_name='Debit Kala Ulang', index=False)
            
            # Sheet 4: Hasil Uji
            test_download_df.to_excel(writer, sheet_name='Hasil Uji', index=False)
            
            # Sheet 5: Parameter Distribusi
            param_data = []
            for dist_name, dist_results in results['results'].items():
                params = dist_results['parameters']
                if dist_name == 'Gumbel':
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'mu0', 'Nilai': params['mu0']})
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'beta', 'Nilai': params['beta']})
                elif dist_name == 'Normal':
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'mu', 'Nilai': params['mu']})
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'sigma', 'Nilai': params['sigma']})
                elif dist_name == 'LogNormal':
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'mu', 'Nilai': params['mu']})
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'sigma', 'Nilai': params['sigma']})
                elif dist_name == 'LogPearson3':
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'mu', 'Nilai': params['mu']})
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'sigma', 'Nilai': params['sigma']})
                    param_data.append({'Distribusi': dist_name, 'Parameter': 'skewness', 'Nilai': params['skew']})
            pd.DataFrame(param_data).to_excel(writer, sheet_name='Parameter Distribusi', index=False)
        
        output.seek(0)
        
        # Tombol download
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.download_button(
                label="📥 Download Excel Lengkap",
                data=output,
                file_name=f"hasil_analisis_banjir_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )
        
        with col_dl2:
            # Download CSV untuk debit kala ulang
            csv = discharge_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Debit",
                data=csv,
                file_name=f"debit_kala_ulang_{timestamp}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col_dl3:
            # Download plot sebagai PNG
            plot_bytes = results['plot_fig'].to_image(format="png")
            st.download_button(
                label="🖼️ Download Plot",
                data=plot_bytes,
                file_name=f"plot_analisis_{timestamp}.png",
                mime="image/png",
                width='stretch'
            )

# Footer
st.markdown("---")
col_footer1, col_footer2 = st.columns([2, 1])
with col_footer1:
    st.markdown("""
    **Aplikasi Analisis Kala Ulang Banjir**  
    Menggunakan metode statistik untuk analisis frekuensi banjir dengan 4 distribusi probabilitas
    """)
with col_footer2:
    st.markdown("""
    **Dikembangkan dengan:**  
    • Streamlit  
    • Plotly  
    • SciPy  
    • NumPy
    """)