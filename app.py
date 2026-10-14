from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import streamlit as st

from model import SYMPTOM_MAP, predict_top_k

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "model.joblib"
HEWAN_LABEL = {"🐄 Sapi": "sapi", "🐃 Kerbau": "kerbau", "🐑 Domba": "domba", "🐐 Kambing": "kambing"}
POEL_OPTIONS = [
    "Belum poel (semua gigi masih gigi susu)",
    "Poel 1 (1 pasang gigi tetap)",
    "Poel 2 (2 pasang gigi tetap)",
    "Poel 3 (3 pasang gigi tetap)",
    "Poel 4 (4 pasang gigi tetap / gigi penuh)",
    "Gigi mulai aus atau tanggal (ternak tua)",
]
# Perkiraan umur (tahun) per tahap poel, metodologi berbeda per jenis ternak:
# - Sapi: poel 1 ~1,5-2 th; poel 2 ~2,5-3 th; poel 3 ~3-3,5 th; poel 4 ~4-4,5 th; aus >8 th.
# - Kerbau: tiap tahap poel +-1 tahun lebih lambat daripada sapi; aus >10 th.
# - Domba: poel 1 ~1-1,5 th; poel 2 ~1,5-2 th; poel 3 ~2,5-3 th; poel 4 ~3,5-4 th; aus >5 th.
# - Kambing: mirip domba tetapi tahap lanjut sedikit lebih lambat (poel 2 ~2-2,5 th; poel 3 ~3-3,5 th).
# Nilai dibulatkan ke tahun representatif agar sesuai kategori umur model.
POEL_TO_AGE = {
    "sapi": [1, 2, 3, 4, 5, 11],
    "kerbau": [2, 3, 4, 5, 6, 12],
    "domba": [1, 2, 2, 3, 4, 8],
    "kambing": [1, 2, 3, 4, 4, 8],
}
GEJALA = sorted(SYMPTOM_MAP.values())
SYMPTOM_INFO = {
    "benjolan tanpa rasa sakit": "Ada tonjolan di kulit atau badan yang tidak membuat ternak kesakitan saat disentuh.",
    "berkeringat": "Badan ternak basah oleh keringat padahal tidak habis bekerja atau kepanasan.",
    "depresi": "Ternak lesu: kepala menunduk, telinga turun, mata sayu, diam memisah dari kawanan, "
               "malas bergerak, tidak tertarik pakan, dan berhenti memamah biak.",
    "kehilangan nafsu makan": "Ternak tidak mau atau hanya sedikit makan, pakan tersisa lebih banyak dari biasanya.",
    "kelelahan": "Ternak tampak lemas dan cepat capek, lebih banyak berbaring, dan tidak kuat berjalan jauh.",
    "kesulitan berjalan": "Ternak berjalan kaku, sempoyongan, atau enggan melangkah.",
    "lepuh pada gusi": "Gelembung kecil berisi cairan pada gusi, seperti melepuh kena air panas.",
    "lepuh pada kuku": "Gelembung berisi cairan di sekitar kuku atau sela kuku, sering membuat ternak pincang.",
    "lepuh pada lidah": "Gelembung berisi cairan di lidah, biasanya membuat ternak susah makan dan banyak mengeluarkan air liur.",
    "lepuh pada mulut": "Gelembung berisi cairan di bibir atau rongga mulut, sering disertai air liur menetes.",
    "luka pada gusi": "Gusi lecet atau berdarah, biasanya bekas lepuh yang pecah.",
    "luka pada kuku": "Kulit di sekitar kuku lecet, terkelupas, atau bernanah sehingga ternak pincang.",
    "luka pada lidah": "Lidah lecet atau terkelupas sehingga ternak kesakitan saat makan.",
    "luka pada mulut": "Bibir atau rongga mulut lecet, biasanya bekas lepuh yang pecah.",
    "menggigil": "Badan ternak gemetar seperti kedinginan walau cuaca tidak dingin.",
    "pembengkakan pada ekstremitas": "Bagian ujung tubuh (kaki bawah, telinga, atau ekor) membesar atau bengkak.",
    "pembengkakan pada leher": "Leher terlihat membesar atau bengkak, kadang membuat susah menelan atau bernapas.",
    "pembengkakan pada otot": "Otot (sering di paha atau bahu) membengkak; kadang terasa seperti ada udara di bawah kulit saat ditekan.",
    "pembengkakan pada perut": "Perut tampak membesar atau membuncit tidak seperti biasanya.",
    "pembengkakan pada tungkai": "Kaki membengkak sehingga ternak susah berdiri atau berjalan.",
    "pincang": "Ternak berjalan timpang, tidak mau menapakkan salah satu kaki.",
    "rasa tidak nyaman di dada": "Ternak gelisah, sering melihat atau menendang ke arah dada, napas tampak tertahan.",
    "sesak napas": "Ternak bernapas cepat dan berat, kadang dengan mulut terbuka atau cuping hidung kembang-kempis.",
    "suara berderak": "Terdengar bunyi kasar seperti kertas diremas saat ternak bernapas.",
}
DISEASE_INFO = {
    "antraks": {
        "penyebab": "Bakteri *Bacillus anthracis*. Sporanya bisa bertahan puluhan tahun di tanah.",
        "penularan": "Ternak memakan rumput/tanah yang tercemar spora. Dapat menular ke manusia "
                     "lewat kontak dengan darah atau daging ternak sakit (zoonosis).",
        "tanda": "Kematian mendadak, demam tinggi, keluar darah kehitaman yang tidak membeku dari "
                 "lubang tubuh (hidung, mulut, anus), bengkak di leher atau perut.",
        "bahaya": "Sangat berbahaya — mematikan untuk ternak dan bisa menular ke manusia. "
                  "Wajib lapor ke dinas peternakan.",
        "penanganan": "Segera hubungi dokter hewan dan dinas peternakan. Jangan menyentuh, membuka, "
                      "memindahkan, menyembelih, atau mengonsumsi bangkai/ternak yang dicurigai. "
                      "Jauhkan manusia dan hewan lain dari lokasi; penanganan bangkai dan pengobatan "
                      "harus dilakukan petugas berwenang.",
        "kata_kunci": "penyakit antraks pada ternak sapi gejala dan penanganan",
    },
    "blackleg": {
        "penyebab": "Bakteri *Clostridium chauvoei* (radang paha). Spora masuk lewat pakan, lalu "
                    "berkembang di otot besar.",
        "penularan": "Tidak menular langsung antarternak; ternak terinfeksi dari spora di tanah "
                     "atau pakan, paling sering menyerang ternak muda yang gemuk.",
        "tanda": "Pincang mendadak, bengkak panas lalu dingin di otot paha/bahu yang berbunyi "
                 "seperti kertas diremas saat ditekan, demam, ternak cepat lemas lalu mati.",
        "bahaya": "Berkembang sangat cepat, sering mematikan dalam 12-48 jam.",
        "penanganan": "Hubungi dokter hewan segera karena penyakit dapat berkembang sangat cepat. "
                      "Batasi pergerakan ternak sakit dan jangan membuka bangkai. Dokter hewan "
                      "menentukan pengobatan, penanganan bangkai, serta vaksinasi pencegahan pada "
                      "ternak sehat.",
        "kata_kunci": "penyakit blackleg radang paha pada sapi gejala dan pengobatan",
    },
    "penyakit mulut dan kuku": {
        "penyebab": "Virus PMK (foot and mouth disease) yang menyerang hewan berkuku belah.",
        "penularan": "Sangat menular: lewat kontak langsung, air liur, udara jarak dekat, serta "
                     "terbawa kendaraan, peralatan, dan alas kaki dari kandang tertular.",
        "tanda": "Lepuh dan luka di mulut, lidah, gusi, dan sela kuku; air liur menetes berlebihan; "
                 "ternak pincang, tidak mau makan, dan demam.",
        "bahaya": "Jarang mematikan pada ternak dewasa tetapi menyebar sangat cepat; wajib lapor "
                  "ke dinas peternakan.",
        "penanganan": "Pisahkan ternak sakit, hentikan lalu lintas ternak dan peralatan, serta "
                      "laporkan kepada dokter hewan/dinas peternakan. Sediakan air bersih dan pakan "
                      "lunak jika ternak mampu menelan. Ikuti arahan petugas untuk disinfeksi, "
                      "perawatan luka, dan vaksinasi; jangan memberikan obat sendiri.",
        "kata_kunci": "penyakit mulut dan kuku PMK pada ternak gejala dan penanganan",
    },
    "virus lumpy": {
        "penyebab": "Virus lumpy skin disease (LSD), kerabat virus cacar, terutama menyerang sapi "
                    "dan kerbau.",
        "penularan": "Terutama lewat gigitan serangga pengisap darah (nyamuk, lalat, caplak), "
                     "juga lewat peralatan dan kontak dengan luka ternak sakit.",
        "tanda": "Benjolan keras di kulit seluruh tubuh (2-5 cm), demam, nafsu makan turun, "
                 "produksi susu turun, kadang benjolan pecah menjadi luka.",
        "bahaya": "Jarang mematikan tetapi menurunkan produksi dan nilai jual ternak; "
                  "penyebarannya cepat lewat serangga.",
        "penanganan": "Pisahkan ternak sakit, batasi perpindahan ternak, dan hubungi dokter hewan/"
                      "dinas peternakan. Kurangi tempat berkembang biak serangga dan jaga kebersihan "
                      "kandang. Perawatan luka, pengendalian serangga, dan vaksinasi perlu mengikuti "
                      "arahan dokter hewan.",
        "kata_kunci": "lumpy skin disease LSD pada sapi gejala dan penanganan",
    },
    "pneumonia": {
        "penyebab": "Peradangan paru-paru akibat bakteri, virus, atau jamur; sering dipicu stres, "
                    "kandang lembap, dingin, atau berdebu.",
        "penularan": "Kuman dapat menyebar lewat udara dan kontak dekat, terutama di kandang padat "
                     "dengan ventilasi buruk.",
        "tanda": "Batuk, napas cepat dan berat, lendir dari hidung, demam, lesu, nafsu makan turun, "
                 "terdengar suara berderak saat bernapas.",
        "bahaya": "Bisa mematikan bila terlambat ditangani, terutama pada ternak muda.",
        "penanganan": "Tempatkan ternak di kandang kering dengan ventilasi baik, kurangi debu dan "
                      "stres, serta sediakan air bersih tanpa memaksa minum. Hubungi dokter hewan "
                      "untuk menentukan penyebab dan pengobatan. Sesak napas berat atau tidak mampu "
                      "berdiri memerlukan pertolongan segera.",
        "kata_kunci": "pneumonia radang paru pada ternak sapi kambing gejala dan pengobatan",
    },
}

st.set_page_config(page_title="Prediksi Penyakit Ternak", page_icon="🐄", layout="centered")
st.markdown(
    """
    <style>
    /* Sembunyikan emblem/branding Streamlit */
    #MainMenu, header[data-testid="stHeader"], footer,
    div[data-testid="stToolbar"], div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    a[class*="viewerBadge"], div[class*="viewerBadge"],
    div[data-testid="stAppDeployButton"] { display: none !important; visibility: hidden !important; }

    .block-container { padding-top: 1.5rem; max-width: 46rem; }
    .stMarkdown p { line-height: 1.65; }
    button, a[data-testid="stLinkButton"] { min-height: 44px; }
    button:focus-visible, a:focus-visible, input:focus-visible {
        outline: 3px solid #14532D !important;
        outline-offset: 3px;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1rem; }
        div[data-testid="stForm"] { padding: 1rem !important; }
        .hero { padding: 1.25rem 1rem !important; }
        .hero h1 { font-size: 1.6rem !important; }
    }
    @media (prefers-reduced-motion: reduce) {
        button { transition: none !important; transform: none !important; }
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #C8E6C9;
        border-radius: 18px;
        padding: 1.75rem;
        box-shadow: 0 6px 24px rgba(46, 125, 50, 0.10);
    }
    div[data-testid="stForm"] h4 {
        color: #2E7D32;
        border-bottom: 2px solid #E8F5E9;
        padding-bottom: 0.35rem;
    }
    div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.85rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #2E7D32, #43A047);
        border: none;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.45);
    }

    .hero {
        background: linear-gradient(135deg, #164E2B, #27653D);
        color: #FFFFFF;
        border-radius: 18px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 28px rgba(27, 94, 32, 0.30);
    }
    .hero .hero-icon { font-size: 3rem; line-height: 1; }
    .hero h1 { color: #FFFFFF; font-size: 2.1rem; margin: 0.35rem 0; padding: 0; letter-spacing: 0.3px; }
    .hero p { margin: 0 auto; font-size: 1.08rem; max-width: 34rem; opacity: 0.95; }

    div[data-testid="stExpander"] details {
        border-radius: 14px;
        border: 1px solid #C8E6C9;
        background: #FFFFFF;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #C8E6C9;
        border-radius: 14px;
        padding: 0.75rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.08);
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: #FFFFFF;
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 0.5rem 0.9rem;
        margin-right: 0.4rem;
    }
    .app-footer {
        text-align: center;
        color: #5B7C62;
        padding: 0.75rem 0 1.25rem 0;
    }
    .app-footer a { color: #2E7D32; font-weight: 600; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def _load_model(mtime: float):
    return joblib.load(MODEL_FILE)

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("model.joblib tidak ditemukan. Jalankan: python train.py")
    # mtime sebagai kunci cache: model baru hasil train.py langsung terpakai tanpa restart.
    return _load_model(MODEL_FILE.stat().st_mtime)

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🐄🐃🐑🐐</div>
        <h1>Prediksi Penyakit Ternak</h1>
        <p>Isi kondisi ternak Anda, sistem memperkirakan penyakit beserta cara penanganan awalnya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "**Bukan rujukan baku.** Hasil aplikasi ini hanya perkiraan untuk edukasi dan penapisan awal, "
    "bukan diagnosis resmi maupun dasar pengambilan keputusan medis. Selalu konsultasikan kondisi "
    "ternak Anda ke ahlinya: dokter hewan, mantri hewan, atau dinas peternakan setempat."
)

st.info(
    "**Cakupan hewan dan batas penyakit yang dapat diprediksi**\n\n"
    "Sistem ini hanya cocok untuk ternak ruminansia: **sapi, kerbau, domba, dan kambing**. "
    "Model dilatih khusus pada data keempat hewan tersebut, sehingga tidak berlaku untuk "
    "unggas, babi, kuda, atau hewan peliharaan lain.\n\n"
    "Model hanya mengenali lima pilihan penyakit:\n"
    "- **Antraks** — infeksi bakteri yang dapat menular ke manusia.\n"
    "- **Blackleg (radang paha)** — infeksi bakteri pada otot.\n"
    "- **Penyakit mulut dan kuku (PMK)** — infeksi virus dengan luka/lepuh di mulut dan kuku.\n"
    "- **Lumpy skin disease (LSD / virus lumpy)** — penyakit benjolan kulit terutama pada sapi dan kerbau.\n"
    "- **Pneumonia (radang paru-paru)** — gangguan paru-paru yang dapat menyebabkan batuk dan sesak napas."
)
with st.expander("Pahami batasan hasil sebelum menggunakan prediksi"):
    st.write(
        "Aplikasi tidak mengenali semua penyakit ternak, misalnya cacingan, kembung, mastitis, "
        "atau keracunan. Model juga tidak memiliki pilihan 'sehat' atau 'penyakit lain'. "
        "Walaupun penyebab sebenarnya berada di luar daftar, model tetap memilih dari lima penyakit tersebut."
    )
    st.write(
        "Tiga hasil teratas adalah kemungkinan alternatif, bukan berarti ternak menderita tiga penyakit "
        "sekaligus. Persentase hanya skor relatif di antara lima kelas model, bukan kepastian diagnosis "
        "atau jaminan bahwa penyakit lain tidak ada. Jenis penyakit yang sesuai juga bergantung pada "
        "jenis hewan; hasil yang tidak sesuai perlu ditinjau dokter hewan."
    )
    st.write(
        "Model menggunakan dataset terbatas dan belum terbukti akurat untuk semua kondisi peternakan. "
        "Gejala yang mirip dapat berasal dari penyakit berbeda. Jangan menunda pemeriksaan dokter hewan "
        "karena hasil aplikasi, terutama bila ternak sesak berat, tidak mampu berdiri, atau mati mendadak."
    )

with st.form("form_prediksi"):
    st.markdown("#### 1️⃣ Pilih jenis hewan")
    hewan_label = st.radio("Jenis hewan", list(HEWAN_LABEL), horizontal=True, label_visibility="collapsed")
    hewan = HEWAN_LABEL[hewan_label]

    st.markdown("#### 2️⃣ Kondisi ternak")
    col1, col2 = st.columns(2)
    with col1:
        poel = st.selectbox("Poel gigi (perkiraan umur)", POEL_OPTIONS, index=2)
        st.warning(
            "Konversi poel ke umur di aplikasi ini masih sementara dan belum divalidasi dokter hewan. "
            "Gigi aus/tanggal tidak cukup untuk menentukan umur pasti; kesalahan umur dapat memengaruhi prediksi."
        )
        st.markdown(
            "[Acuan FAO, Unit 9: pendugaan umur dari gigi](https://www.fao.org/4/t0690e/t0690e05.htm). "
            "Acuan membedakan sapi dan kerbau, tetapi mengelompokkan domba dan kambing. "
            "Angka konversi aplikasi belum boleh dianggap tabel umur resmi."
        )
        st.caption(
            "Lihat gigi seri bawah ternak: hitung berapa pasang gigi besar (gigi tetap) "
            "yang sudah mengganti gigi susu. Perkiraan umur berbeda untuk sapi, kerbau, "
            "domba, dan kambing — kerbau poel lebih lambat, domba/kambing lebih cepat. "
            "Hasilnya perkiraan, bukan umur pasti; dipengaruhi ras, pakan, dan lingkungan."
        )
    with col2:
        suhu = st.number_input("Suhu tubuh (°C)", min_value=32.3, max_value=43.3, value=38.9, step=0.1)
        st.caption("Suhu normal sapi/kerbau/domba/kambing sekitar 38,0–39,5 °C.")

    st.markdown("#### 3️⃣ Pilih gejala yang terlihat (1 sampai 3 gejala)")
    gejala_terpilih = st.multiselect(
        "Gejala",
        GEJALA,
        default=["kehilangan nafsu makan"],
        max_selections=3,
        label_visibility="collapsed",
        placeholder="Pilih 1-3 gejala",
    )
    st.caption("Semakin lengkap gejala yang dipilih, semakin akurat perkiraan penyakit.")
    with st.expander("ℹ️ Keterangan semua gejala"):
        for gejala in GEJALA:
            st.markdown(f"**{gejala.capitalize()}** — {SYMPTOM_INFO[gejala]}")
    submitted = st.form_submit_button("🔍 Prediksi penyakit", type="primary", use_container_width=True)

if submitted:
    if not gejala_terpilih:
        st.error("Pilih minimal satu gejala.")
    else:
        usia = POEL_TO_AGE[hewan][POEL_OPTIONS.index(poel)]
        gejala_isi = gejala_terpilih + [None] * (3 - len(gejala_terpilih))
        input_data = pd.DataFrame([{
            "Hewan": hewan, "Usia": usia, "Suhu": suhu * 9 / 5 + 32,
            "Gejala 1": gejala_isi[0], "Gejala 2": gejala_isi[1], "Gejala 3": gejala_isi[2],
        }])
        try:
            results = predict_top_k(load_model(), input_data, 3)
        except Exception as exc:
            st.error(f"Model gagal dijalankan: {exc}")
        else:
            st.subheader("📋 Hasil prediksi")
            st.success(f"Kemungkinan utama: **{results[0][0].title()}** ({results[0][1]:.1%})")
            metric_cols = st.columns(len(results))
            for col, (disease, probability) in zip(metric_cols, results):
                col.metric(disease.title(), f"{probability:.1%}")
            chart = pd.DataFrame(results, columns=["Penyakit", "Probabilitas"]).set_index("Penyakit")
            st.bar_chart(chart, color="#2E7D32")
            st.caption("Probabilitas menunjukkan keyakinan relatif model, bukan kepastian diagnosis klinis.")
            st.info(
                "Jadikan hasil ini sebagai bahan awal saat berkonsultasi — bukan rujukan baku. "
                "Pemeriksaan langsung oleh dokter hewan tetap diperlukan untuk memastikan diagnosis."
            )
            st.subheader("🩺 Informasi penyakit dan penanganan awal")
            st.warning("Panduan berikut berlaku jika penyakit dicurigai, bukan diagnosis pasti. "
                       "Jangan memberikan antibiotik atau obat lain tanpa arahan dokter hewan.")
            for disease, probability in results:
                with st.expander(f"{disease.title()} — {probability:.1%}", expanded=disease == results[0][0]):
                    info = DISEASE_INFO.get(disease)
                    if info is None:
                        st.write("Informasi belum tersedia. Hubungi dokter hewan untuk pemeriksaan.")
                    else:
                        st.markdown(f"**Penyebab** — {info['penyebab']}")
                        st.markdown(f"**Cara penularan** — {info['penularan']}")
                        st.markdown(f"**Tanda-tanda** — {info['tanda']}")
                        st.markdown(f"**Tingkat bahaya** — {info['bahaya']}")
                        st.markdown("**Upaya penanganan awal**")
                        st.write(info["penanganan"])
                        search_url = "https://www.google.com/search?q=" + quote_plus(info["kata_kunci"])
                        st.link_button(f"🔎 Cari info lebih lanjut tentang {disease}", search_url)

            st.subheader("Lanjutkan bertanya kepada AI")
            prompt = (
                "Bantu saya memahami kondisi ternak berikut dalam bahasa Indonesia yang mudah dipahami peternak.\n\n"
                f"Jenis ternak: {hewan}\n"
                f"Kondisi gigi: {poel}\n"
                f"Umur yang digunakan model: {usia} tahun (konversi poel sementara, belum divalidasi dokter hewan).\n"
                f"Suhu tubuh: {suhu:.1f} °C\n"
                f"Gejala yang dilaporkan: {', '.join(gejala_terpilih)}\n"
                "Gejala yang tidak dipilih belum diketahui, bukan berarti tidak ada.\n\n"
                "Hasil perkiraan model:\n"
                + "\n".join(f"- {disease}: {probability:.1%}" for disease, probability in results)
                + "\n\nModel hanya mengenali antraks, blackleg, PMK, LSD, dan pneumonia; "
                "tidak memiliki kelas sehat atau penyakit lain. Skor bukan kepastian diagnosis. "
                "Ini dataset terbatas, bukan validasi klinis.\n\n"
                "Jelaskan kemungkinan penyebab dan keterbatasannya tanpa memastikan diagnosis. "
                "Jangan menganggap semua penyakit hasil model terjadi bersamaan. "
                "Jelaskan tanda darurat dan tindakan awal yang aman sambil menghubungi dokter hewan. "
                "Jangan memberikan resep, dosis obat, atau prosedur invasif. "
                "Jangan mengarang durasi sakit, riwayat vaksinasi, atau informasi yang belum diberikan. "
                "Tanyakan satu informasi tambahan paling penting terlebih dahulu. "
                "Gunakan data ini sebagai konteks untuk pertanyaan lanjutan dalam percakapan yang sama; "
                "perbarui ringkasan bila saya memberikan informasi baru."
            )
            st.write("Salin teks melalui tombol salin pada kotak berikut, lalu tempel ke AI.")
            with st.expander("Lihat dan salin prompt konsultasi AI"):
                st.code(prompt, language=None, wrap_lines=True)
            st.link_button("Buka AI untuk konsultasi informasi", "https://www.adioranye.my.id/ai")
            st.caption(
                "Lanjutkan pertanyaan di percakapan AI yang sama agar konteks tetap tersambung. "
                "Data tidak dikirim otomatis. Jika membuka percakapan baru, tempel kembali prompt "
                "beserta ringkasan jawaban sebelumnya. AI bukan pengganti dokter hewan."
            )

with st.expander("Tentang model"):
    st.write(
        "Model menggunakan Random Forest dengan one-hot encoding. Data berisi sapi, kerbau, "
        "domba, dan kambing serta lima kelas penyakit: antraks, blackleg, penyakit mulut dan "
        "kuku, virus lumpy, dan pneumonia. Seluruh masukan dan keluaran telah diterjemahkan "
        "ke Bahasa Indonesia."
    )
    st.write(
        "Pelatihan model dilakukan menggunakan dataset dari Kaggle: "
        "[Livestock Disease Diagnosis Dataset]"
        "(https://www.kaggle.com/datasets/captaingee/livestock-disease-diagnosis-dataset), "
        "yang berisi lebih dari 43.000 catatan gejala dan diagnosis penyakit ternak."
    )

st.divider()
st.markdown(
    f"<div class='app-footer'>© {datetime.now(ZoneInfo('Asia/Jakarta')).year} · Developed by "
    "<a href='https://www.adioranye.my.id'>Galuh Adi Insani</a></div>",
    unsafe_allow_html=True,
)
