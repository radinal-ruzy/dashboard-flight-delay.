import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ----------------------------
# 🎨 Tampilan dasar & CSS
# ----------------------------
st.set_page_config(page_title="Dashboard Keterlambatan Pesawat", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #1f77b4;
    }
    h2, h3, h4 {
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 📌 Sidebar
# ----------------------------
st.sidebar.title("ℹ️ Tentang Dashboard")
st.sidebar.markdown("""
**Dataset:** Flight Delay  
**Tujuan:** Prediksi `late_aircraft_delay` menggunakan regresi linier.  
**Pembuat:** ✨ Kamu sendiri  
---
Gunakan **tab di atas** untuk berpindah antar-analisis.
""")

# ----------------------------
# 📂 Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("flight_delay_cleaned.csv")  # ganti dengan nama file CSV kamu
    if 'carrier' in df.columns and 'airport' in df.columns:
        df = df.drop(columns=['carrier', 'airport'])
    return df

df = load_data()

# ----------------------------
# 🏷️ Judul utama
# ----------------------------
st.title("✈️ Dashboard Penelitian Keterlambatan Pesawat")
st.markdown("""
Selamat datang di dashboard penelitian **keterlambatan pesawat**.  
Data ini dianalisis untuk memprediksi keterlambatan lanjutan (`late_aircraft_delay`) menggunakan regresi linier.
---
""")

# ----------------------------
# 📊 Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Korelasi",
    "🤖 Regresi Linier",
    "✅ Evaluasi Model"
])

# ========== TAB 1 ==========
with tab1:
    st.subheader("📊 Eksplorasi Data & Proporsi Delay")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"Dataset memiliki **{df.shape[0]} baris** dan **{df.shape[1]} kolom**")
        st.markdown("#### Contoh Data:")
        st.dataframe(df.head())

    with c2:
        st.markdown("#### Proporsi total delay per kategori:")
        delay_cols = ['carrier_delay','weather_delay','nas_delay','late_aircraft_delay']
        sums = df[delay_cols].sum()

        fig1, ax1 = plt.subplots()
        ax1.pie(sums, labels=delay_cols, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

# ========== TAB 2 ==========
with tab2:
    st.subheader("📈 Analisis Korelasi")
    st.markdown("Heatmap korelasi antara variabel-variabel keterlambatan.")

    delay_cols = ['carrier_delay','weather_delay','nas_delay','late_aircraft_delay']
    corr = df[delay_cols].corr()

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.write("""
    🔎 **Insight:**  
    - `late_aircraft_delay` paling berkorelasi dengan `carrier_delay` dan `nas_delay`.  
    - `weather_delay` kontribusinya lebih kecil.
    """)

# --- TAB 3: REGRESI ---
with tab3:
    st.subheader("🤖 Membuat Model Regresi Linier")

    # Target & fitur (drop year & month)
    y = df["late_aircraft_delay"]
    X = df.drop(columns=[
        "late_aircraft_delay", "arr_delay",
        "carrier_delay", "weather_delay", "nas_delay", "security_delay",
        "year", "month"
    ])

    # Bagi data train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Latih model
    model = LinearRegression()
    model.fit(X_train, y_train)

    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Koefisien"])
    st.write("### Koefisien Regresi Linier")
    st.dataframe(coeff_df.style.background_gradient(cmap="Blues"))

    st.write(f"**Intercept:** `{model.intercept_:.4f}`")

# --- TAB 4: EVALUASI ---
with tab4:
    st.subheader("✅ Evaluasi Model")

    # Prediksi & evaluasi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 🔎 Ringkasan Metrik")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f} mnt")
    m2.metric("RMSE", f"{rmse:.2f} mnt")
    m3.metric("R²", f"{r2:.4f}")

    st.markdown("### 📈 Prediksi vs Aktual")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.3, color="blue", label="Prediksi")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y=x)")
    ax3.set_xlabel("Aktual late_aircraft_delay")
    ax3.set_ylabel("Prediksi late_aircraft_delay")
    ax3.set_title("Prediksi vs Aktual (Test Set)")
    ax3.legend()
    st.pyplot(fig3)

    st.write("""
    ✨ **Insight:**  
    Model regresi linier mampu menjelaskan sebagian besar variasi data dengan baik.
    """)

st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit | Dataset: flight_delay_cleaned.csv")
