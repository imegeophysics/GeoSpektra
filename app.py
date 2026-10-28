import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from folium.raster_layers import ImageOverlay
from streamlit_folium import folium_static
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS STYLING
# ==========================================
st.set_page_config(
    page_title="GeoSpectra Pro - Dasbor Geoteknik & Gempa",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_style = "<style>.main {background-color: #f8f9fa;} .metric-card {background-color: #ffffff; border-radius: 10px; padding: 15px 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border-left: 5px solid #1f77b4; margin-bottom: 10px;} .metric-title {font-size: 0.82rem; color: #6c757d; font-weight: 600; text-transform: uppercase;} .metric-value {font-size: 1.45rem; font-weight: 700; color: #212529;} .stTabs [data-baseweb='tab-list'] {gap: 10px;} .stTabs [data-baseweb='tab'] {background-color: #ffffff; border-radius: 8px 8px 0px 0px; padding: 10px 20px; font-weight: 600;}</style>"
st.markdown(css_style, unsafe_allow_html=True)

# ==========================================
# 2. SISTEM AUTENTIKASI (USER: mitahir | PASS: mitahir1980)
# ==========================================
USER_DATABASE = {
    "mitahir": "mitahir1980"
}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login_form():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1.2, 1])
    with col_b:
        st.markdown(
            "<div style='background-color:#ffffff; padding:30px; border-radius:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align:center;'>"
            "<img src='https://img.icons8.com/color/96/earthquake.png' width='80'>"
            "<h2 style='margin-top:10px;'>GeoSpectra Pro</h2>"
            "<p style='color:#6c757d;'>Masuk untuk Mengakses Dasbor Geoteknik</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            user_input = st.text_input("Username:")
            pass_input = st.text_input("Password:", type="password")
            submit_button = st.form_submit_button("🔑 Masuk Ke Aplikasi", use_container_width=True)
            
            if submit_button:
                if user_input in USER_DATABASE and USER_DATABASE[user_input] == pass_input:
                    st.session_state["authenticated"] = True
                    st.session_state["logged_user"] = user_input
                    st.rerun()
                else:
                    st.error("Username atau Password yang Anda masukkan salah!")

if not st.session_state["authenticated"]:
    login_form()
    st.stop()

# ==========================================
# 3. FUNGSI UTILITY & KALKULASI
# ==========================================
@st.cache_data
def load_default_data():
    try:
        df = pd.read_excel('Vs30_Makassar.xlsx')
        if 'Kecamatan' not in df.columns:
            df['Kecamatan'] = 'Kota Makassar'
        df['Point_ID'] = [f"Titik {i+1} ({row['Kecamatan']} | Lat: {row['Lat']:.4f}, Long: {row['Long']:.4f})" for i, row in df.iterrows()]
        return df
    except Exception:
        return pd.DataFrame(columns=['Lat', 'Long', 'Vs30m/s', 'Site Class', 'Kecamatan', 'Point_ID'])

def hitung_respon_spektra(Ss, S1, site_class, Ie=1.0):
    ss_keys = [0.25, 0.50, 0.75, 1.00, 1.25]
    fa_table = {
        'A': [0.8, 0.8, 0.8, 0.8, 0.8],
        'B': [0.9, 0.9, 0.9, 0.9, 0.9],
        'C': [1.3, 1.3, 1.2, 1.2, 1.2],
        'D': [1.6, 1.4, 1.2, 1.1, 1.0],
        'E': [2.4, 1.7, 1.3, 1.0, 0.9]
    }
    s1_keys = [0.10, 0.20, 0.30, 0.40, 0.50]
    fv_table = {
        'A': [0.8, 0.8, 0.8, 0.8, 0.8],
        'B': [0.8, 0.8, 0.8, 0.8, 0.8],
        'C': [1.5, 1.5, 1.5, 1.4, 1.3],
        'D': [2.4, 2.2, 2.0, 1.8, 1.6],
        'E': [4.2, 3.3, 2.8, 2.4, 2.2]
    }
    sc = str(site_class).strip().upper()
    if sc not in fa_table:
        sc = 'D'
    Fa = float(np.interp(Ss, ss_keys, fa_table[sc]))
    Fv = float(np.interp(S1, s1_keys, fv_table[sc]))
    Sms = Fa * Ss
    Sm1 = Fv * S1
    Sds = (2.0 / 3.0) * Sms
    Sd1 = (2.0 / 3.0) * Sm1
    T0 = 0.2 * (Sd1 / Sds) if Sds > 0 else 0.04
    Ts = Sd1 / Sds if Sds > 0 else 0.2
    T = np.linspace(0.0, 4.0, 500)
    Sa = []
    for t in T:
        if t < T0:
            sa = Sds * (0.4 + 0.6 * (t / T0))
        elif T0 <= t <= Ts:
            sa = Sds
        else:
            sa = Sd1 / t
        Sa.append(sa * Ie)
    return T, Sa, Fa, Fv, Sds, Sd1, T0, Ts

def estimasi_titik_terdekat(user_lat, user_long, df):
    if df.empty:
        return None, 0
    df_calc = df.copy()
    df_calc['dist'] = np.sqrt((df_calc['Lat'] - user_lat)**2 + (df_calc['Long'] - user_long)**2)
    closest = df_calc.loc[df_calc['dist'].idxmin()]
    dist_km = closest['dist'] * 111.0
    return closest, dist_km

def make_card(title, value, color_hex):
    return (
        '<div class="metric-card" style="border-left-color: ' + color_hex + ';">'
        '<div class="metric-title">' + title + '</div>'
        '<div class="metric-value">' + str(value) + '</div>'
        '</div>'
    )

# ==========================================
# 4. CONTROLS BILAH SAMPING (SIDEBAR)
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/earthquake.png", width=64)
st.sidebar.title("GeoSpectra Pro")
st.sidebar.write("Pengguna: **" + str(st.session_state.get("logged_user", "Pengguna")) + "**")

if st.sidebar.button("🚪 Keluar (Logout)", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📁 1. Sumber Data")
uploaded_file = st.sidebar.file_uploader("Unggah File Excel/CSV Mandiri:", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        if 'Kecamatan' not in df.columns:
            df['Kecamatan'] = 'Kota Makassar'
        df['Point_ID'] = [f"Titik {i+1} ({row['Kecamatan']} | Lat: {row['Lat']:.4f}, Long: {row['Long']:.4f})" for i, row in df.iterrows()]
        st.sidebar.success("Dataset berhasil dimuat!")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat file: {e}")
        df = load_default_data()
else:
    df = load_default_data()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 2. Filter Spasial & Layer")

all_kec = sorted(df['Kecamatan'].unique()) if 'Kecamatan' in df.columns else ['Kota Makassar']
selected_kec = st.sidebar.multiselect("Filter Kecamatan:", options=all_kec, default=all_kec)

selected_classes = st.sidebar.multiselect(
    "Filter Kelas Tanah:",
    options=sorted(df['Site Class'].unique()) if not df.empty else [],
    default=sorted(df['Site Class'].unique()) if not df.empty else []
)

if not df.empty:
    min_vs, max_vs = float(df['Vs30m/s'].min()), float(df['Vs30m/s'].max())
    selected_vs_range = st.sidebar.slider("Rentang Vs30 (m/s):", min_value=min_vs, max_value=max_vs, value=(min_vs, max_vs))
    filtered_df = df[
        (df['Kecamatan'].isin(selected_kec)) &
        (df['Site Class'].isin(selected_classes)) & 
        (df['Vs30m/s'].between(selected_vs_range[0], selected_vs_range[1]))
    ]
else:
    filtered_df = df.copy()

show_heatmap = st.sidebar.checkbox("🔥 Tampilkan Layer Heatmap", value=False)
show_interp = st.sidebar.checkbox("🌈 Tampilkan Layer Interpolasi Vs30", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🌋 3. Parameter Gempa PUPR")
input_Ss = st.sidebar.number_input("Nilai Ss (g):", value=1.10, step=0.05)
input_S1 = st.sidebar.number_input("Nilai S1 (g):", value=0.45, step=0.05)

kategori_risiko = st.sidebar.selectbox(
    "Kategori Risiko Bangunan:",
    ["Kategori I & II (Gedung Umum/Perumahan)", 
     "Kategori III (Fasilitas Publik/Sekolah)", 
     "Kategori IV (Fasilitas Esensial/Rumah Sakit)"]
)

map_Ie = {
    "Kategori I & II (Gedung Umum/Perumahan)": 1.0,
    "Kategori III (Fasilitas Publik/Sekolah)": 1.25,
    "Kategori IV (Fasilitas Esensial/Rumah Sakit)": 1.50
}
selected_Ie = map_Ie[kategori_risiko]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 4. Pengecekan Titik Proyek")
use_custom_point = st.sidebar.checkbox("Input Koordinat Proyek Kustom", value=False)

if use_custom_point:
    c_lat = st.sidebar.number_input("Latitude Proyek:", value=-5.1476, format="%.4f")
    c_long = st.sidebar.number_input("Longitude Proyek:", value=119.4327, format="%.4f")
else:
    c_lat, c_long = None, None

st.sidebar.markdown("---")
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Unduh Data Terfilter (CSV)",
    data=csv_data,
    file_name="data_vs30_geospectra.csv",
    mime="text/csv"
)

# ==========================================
# 5. TAMPILAN UTAMA DASBOR
# ==========================================
st.title("🏙️ Dasbor Geoteknik & Respon Spektra Interaktif")
st.markdown("Integrasi data klasifikasi tanah Vs30 dengan Peta Google Maps dan Perhitungan Beban Gempa Spektra **SNI 1726:2019 / RSA PUPR**.")

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.markdown(make_card("Total Titik Data", len(filtered_df), "#007bff"), unsafe_allow_html=True)

with col_m2:
    mean_vs = f"{filtered_df['Vs30m/s'].mean():.1f} m/s" if not filtered_df.empty else "-"
    st.markdown(make_card("Vs30 Rata-Rata", mean_vs, "#28a745"), unsafe_allow_html=True)

with col_m3:
    dom_class = filtered_df['Site Class'].mode()[0] if not filtered_df.empty else "-"
    st.markdown(make_card("Kelas Dominan", "Kelas " + str(dom_class), "#ffc107"), unsafe_allow_html=True)

with col_m4:
    param_str = "Ss:" + str(input_Ss) + " | S1:" + str(input_S1)
    st.markdown(make_card("Parameter Gempa", param_str, "#dc3545"), unsafe_allow_html=True)

with col_m5:
    st.markdown(make_card("Faktor Keutamaan (Ie)", f"{selected_Ie:.2f}", "#6f42c1"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("📍 Peta Interaktif Sebaran Kelas Tanah")

center_lat = filtered_df['Lat'].mean() if not filtered_df.empty else -5.1476
center_long = filtered_df['Long'].mean() if not filtered_df.empty else 119.4327

m = folium.Map(location=[center_lat, center_long], zoom_start=12)

folium.TileLayer('http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}', attr='Google', name='Google Street').add_to(m)
folium.TileLayer('http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)

color_map = {'B': 'purple', 'C': 'green', 'D': 'orange', 'E': 'red'}

for _, row in filtered_df.iterrows():
    kec_name = row.get('Kecamatan', 'Makassar')
    popup_text = (
        "<div style='font-family: Arial; width: 180px;'>"
        "<h5 style='margin:0;'>Kelas " + str(row['Site Class']) + "</h5>"
        "<b>Kecamatan:</b> " + str(kec_name) + "<br>"
        "<b>Vs30:</b> " + str(row['Vs30m/s']) + " m/s<br>"
        "<b>Lat:</b> " + f"{row['Lat']:.4f}" + "<br>"
        "<b>Long:</b> " + f"{row['Long']:.4f}" + "</div>"
    )
    folium.CircleMarker(
        location=[row['Lat'], row['Long']],
        radius=6,
        popup=folium.Popup(popup_text, max_width=250),
        tooltip="[" + str(kec_name) + "] Kelas " + str(row['Site Class']) + " | Vs30: " + str(row['Vs30m/s']) + " m/s",
        color="black",
        weight=1,
        fill=True,
        fill_color=color_map.get(row['Site Class'], 'blue'),
        fill_opacity=0.85
    ).add_to(m)

if use_custom_point and c_lat and c_long:
    folium.Marker(
        location=[c_lat, c_long],
        popup="Lokasi Proyek Kustom",
        tooltip="📍 Titik Proyek",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)

if show_heatmap and not filtered_df.empty:
    heat_data = [[row['Lat'], row['Long'], row['Vs30m/s']] for _, row in filtered_df.iterrows()]
    HeatMap(heat_data, radius=20, blur=15).add_to(m)

if show_interp and len(filtered_df) >= 4:
    lats = filtered_df['Lat'].values
    longs = filtered_df['Long'].values
    vs30 = filtered_df['Vs30m/s'].values
    grid_lat = np.linspace(lats.max(), lats.min(), 150)
    grid_long = np.linspace(longs.min(), longs.max(), 150)
    grid_x, grid_y = np.meshgrid(grid_long, grid_lat)
    grid_z = griddata((longs, lats), vs30, (grid_x, grid_y), method='linear')
    norm = mcolors.Normalize(vmin=np.nanmin(vs30), vmax=np.nanmax(vs30))
    try:
        cmap = matplotlib.colormaps['turbo']
    except AttributeError:
        cmap = plt.get_cmap('turbo')
    rgba_img = cmap(norm(grid_z))
    rgba_img[np.isnan(grid_z)] = [0, 0, 0, 0]
    ImageOverlay(
        image=rgba_img,
        bounds=[[lats.min(), longs.min()], [lats.max(), longs.max()]],
        opacity=0.45,
        name="Interpolasi Kontinu Vs30"
    ).add_to(m)

folium.LayerControl().add_to(m)
folium_static(m, width=1150, height=520)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Respon Spektra Desain", 
    "🎯 Estimasi Titik Proyek Kustom", 
    "📊 Statistik & Distribusi Spasial", 
    "📄 Tabel Data & Ekspor"
])

with tab1:
    col_t1_left, col_t1_right = st.columns([1, 3])
    with col_t1_left:
        st.markdown("#### Pengaturan Grafik")
        mode_spektra = st.radio(
            "Pilih Tampilan Kurva:",
            ["Perbandingan Seluruh Kelas Tanah", "Titik Spesifik Lapangan"],
            index=0
        )
        if mode_spektra == "Titik Spesifik Lapangan" and not filtered_df.empty:
            selected_point_str = st.selectbox("Pilih Titik Lokasi:", options=filtered_df['Point_ID'].tolist())
            point_data = filtered_df[filtered_df['Point_ID'] == selected_point_str].iloc[0]
            target_class = point_data['Site Class']
            target_vs = point_data['Vs30m/s']
        else:
            target_class = 'D'
            target_vs = None

    with col_t1_right:
        fig_spectra = go.Figure()
        if mode_spektra == "Perbandingan Seluruh Kelas Tanah":
            available_classes = sorted(filtered_df['Site Class'].unique()) if not filtered_df.empty else ['B','C','D','E']
            for sc in available_classes:
                T_val, Sa_val, Fa_val, Fv_val, Sds_val, Sd1_val, T0_val, Ts_val = hitung_respon_spektra(
                    input_Ss, input_S1, sc, Ie=selected_Ie
                )
                fig_spectra.add_trace(go.Scatter(
                    x=T_val, y=Sa_val, mode='lines',
                    name=f"Kelas Tanah {sc} (SDS={Sds_val:.3f}g)",
                    line=dict(color=color_map.get(sc, 'blue'), width=2.5)
                ))
        else:
            T_val, Sa_val, Fa_val, Fv_val, Sds_val, Sd1_val, T0_val, Ts_val = hitung_respon_spektra(
                input_Ss, input_S1, target_class, Ie=selected_Ie
            )
            fig_spectra.add_trace(go.Scatter(
                x=T_val, y=Sa_val, mode='lines',
                name=f"Titik Terpilih - Kelas {target_class}",
                line=dict(color=color_map.get(target_class, 'red'), width=3.5)
            ))
            info_msg = f"**Parameter Desain Titik:** Fa = **{Fa_val:.2f}** | Fv = **{Fv_val:.2f}** | SDS = **{Sds_val:.3f}g** | SD1 = **{Sd1_val:.3f}g** | Ts = **{Ts_val:.3f} s**"
            st.info(info_msg)

        fig_spectra.update_layout(
            title=f"Kurva Respon Spektra Desain (SNI 1726:2019) - Faktor Keutamaan Ie = {selected_Ie}",
            xaxis_title="Periode Struktural T (detik)",
            yaxis_title="Percepatan Spektra Desain Sa (g)",
            template="plotly_white",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig_spectra, use_container_width=True)

with tab2:
    if use_custom_point and c_lat and c_long:
        st.markdown("#### 🎯 Hasil Interpolasi & Estimasi Lokasi Proyek Kustom")
        closest_point, dist_km = estimasi_titik_terdekat(c_lat, c_long, df)
        if closest_point is not None:
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Koordinat Proyek", f"{c_lat:.4f}, {c_long:.4f}")
            ec2.metric("Titik Acuan Terdekat", f"{dist_km:.2f} km")
            ec3.metric("Estimasi Vs30", f"{closest_point['Vs30m/s']} m/s")
            ec4.metric("Estimasi Kelas Tanah", f"Kelas {closest_point['Site Class']}")
            st.markdown("---")
            T_c, Sa_c, Fa_c, Fv_c, Sds_c, Sd1_c, T0_c, Ts_c = hitung_respon_spektra(
                input_Ss, input_S1, closest_point['Site Class'], Ie=selected_Ie
            )
            fig_custom = go.Figure()
            fig_custom.add_trace(go.Scatter(
                x=T_c, y=Sa_c, mode='lines',
                name=f"Estimasi Spektra Proyek (Kelas {closest_point['Site Class']})",
                line=dict(color='darkred', width=3)
            ))
            fig_custom.update_layout(
                title="Kurva Respon Spektra Desain untuk Lokasi Proyek Spesifik",
                xaxis_title="Periode T (detik)", yaxis_title="Sa (g)", template="plotly_white"
            )
            st.plotly_chart(fig_custom, use_container_width=True)
    else:
        st.info("💡 Silakan aktifkan centang **'Input Koordinat Proyek Kustom'** di sidebar sebelah kiri untuk menggunakan fitur analisis titik proyek spesifik ini.")

with tab3:
    if not filtered_df.empty:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_hist = px.histogram(
                filtered_df, x="Vs30m/s", color="Site Class", marginal="box",
                title="Sebaran Frekuensi Nilai Vs30",
                color_discrete_map=color_map, template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_s2:
            fig_pie = px.pie(
                filtered_df, names="Site Class", title="Proporsi Persentase Kelas Tanah",
                color="Site Class", color_discrete_map=color_map, hole=0.4
            )
            fig_pie.update_layout(template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        if 'Kecamatan' in filtered_df.columns and filtered_df['Kecamatan'].nunique() > 1:
            st.markdown("---")
            fig_kec = px.bar(
                filtered_df.groupby(['Kecamatan', 'Site Class']).size().reset_index(name='Jumlah Titik'),
                x="Kecamatan", y="Jumlah Titik", color="Site Class",
                title="Jumlah Titik Pengujian per Kecamatan",
                color_discrete_map=color_map, template="plotly_white"
            )
            st.plotly_chart(fig_kec, use_container_width=True)

with tab4:
    st.markdown("#### 📄 Seluruh Data Geoteknik Terfilter")
    st.dataframe(filtered_df, use_container_width=True)