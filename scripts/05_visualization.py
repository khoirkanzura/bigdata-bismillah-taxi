import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Konfigurasi Path
# ============================================================

BASE_DIR = "/home/nadya/bigdata-bismillah-taxi"
INPUT_DIR = f"{BASE_DIR}/output/final"
OUTPUT_DIR = f"{BASE_DIR}/output/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Konfigurasi Tampilan Grafik
# ============================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})


def format_int(value):
    return f"{int(value):,}".replace(",", ".")


def format_float(value, digits=4):
    return f"{float(value):.{digits}f}"


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight")
    plt.close()


# ============================================================
# 3. Membaca File Hasil Eksperimen
# ============================================================

data_summary = pd.read_csv(f"{INPUT_DIR}/data_summary.csv")
evaluation_result = pd.read_csv(f"{INPUT_DIR}/evaluation_result.csv")
feature_importance = pd.read_csv(f"{INPUT_DIR}/feature_importance.csv")
prediction_sample = pd.read_csv(f"{INPUT_DIR}/prediction_sample.csv")


# ============================================================
# 4. Visualisasi Jumlah Data Setelah Preprocessing
# ============================================================

label_map = {
    "Jumlah data awal": "Data awal",
    "Jumlah data setelah dropna": "Setelah dropna",
    "Jumlah data setelah filter dasar": "Filter dasar",
    "Jumlah data setelah filter jarak": "Filter jarak",
    "Jumlah data final setelah preprocessing": "Data final"
}

data_summary["tahap"] = data_summary["keterangan"].map(label_map).fillna(data_summary["keterangan"])

plt.figure(figsize=(9, 5))
bars = plt.barh(data_summary["tahap"], data_summary["jumlah"])

plt.title("Jumlah Data pada Setiap Tahap Preprocessing")
plt.xlabel("Jumlah data")
plt.ylabel("Tahap preprocessing")
plt.ticklabel_format(style="plain", axis="x")

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + (data_summary["jumlah"].max() * 0.005),
        bar.get_y() + bar.get_height() / 2,
        format_int(width),
        va="center"
    )

save_figure("01_data_summary.png")


# ============================================================
# 5. Visualisasi Perbandingan RMSE
# ============================================================

evaluation_sorted_rmse = evaluation_result.sort_values("rmse", ascending=True)

plt.figure(figsize=(8, 5))
bars = plt.bar(evaluation_sorted_rmse["model"], evaluation_sorted_rmse["rmse"])

plt.title("Perbandingan RMSE Model Regresi")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=15, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        format_float(height, 3),
        ha="center",
        va="bottom"
    )

save_figure("02_model_rmse.png")


# ============================================================
# 6. Visualisasi Perbandingan MAE
# ============================================================

evaluation_sorted_mae = evaluation_result.sort_values("mae", ascending=True)

plt.figure(figsize=(8, 5))
bars = plt.bar(evaluation_sorted_mae["model"], evaluation_sorted_mae["mae"])

plt.title("Perbandingan MAE Model Regresi")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=15, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        format_float(height, 3),
        ha="center",
        va="bottom"
    )

save_figure("03_model_mae.png")


# ============================================================
# 7. Visualisasi Perbandingan R2
# ============================================================

evaluation_sorted_r2 = evaluation_result.sort_values("r2", ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(evaluation_sorted_r2["model"], evaluation_sorted_r2["r2"])

plt.title("Perbandingan R² Model Regresi")
plt.xlabel("Model")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        format_float(height, 3),
        ha="center",
        va="bottom"
    )

save_figure("04_model_r2.png")


# ============================================================
# 8. Visualisasi Feature Importance
# ============================================================

feature_sorted = feature_importance.sort_values("importance", ascending=True)

plt.figure(figsize=(9, 6))
bars = plt.barh(feature_sorted["feature"], feature_sorted["importance"])

plt.title("Feature Importance Model Terbaik")
plt.xlabel("Importance")
plt.ylabel("Fitur")

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        format_float(width, 4),
        va="center"
    )

save_figure("05_feature_importance.png")


# ============================================================
# 9. Visualisasi Distribusi Error Prediksi
# ============================================================

prediction_sample["absolute_error"] = (
    prediction_sample["label"] - prediction_sample["prediction"]
).abs()

plt.figure(figsize=(8, 5))
plt.hist(prediction_sample["absolute_error"], bins=30)

plt.title("Distribusi Absolute Error pada Sampel Prediksi")
plt.xlabel("Absolute error")
plt.ylabel("Frekuensi")

save_figure("06_prediction_error_distribution.png")


# ============================================================
# 10. Visualisasi Actual vs Prediction
# Dibuat hanya sebagai gambar pendukung, bukan gambar utama artikel.
# ============================================================

plt.figure(figsize=(7, 6))
plt.scatter(
    prediction_sample["label"],
    prediction_sample["prediction"],
    alpha=0.5
)

max_value = max(
    prediction_sample["label"].max(),
    prediction_sample["prediction"].max()
)

plt.plot([0, max_value], [0, max_value], linestyle="--")

plt.title("Perbandingan Tarif Aktual dan Tarif Prediksi")
plt.xlabel("Tarif aktual")
plt.ylabel("Tarif prediksi")

save_figure("07_actual_vs_prediction_supporting.png")


# ============================================================
# 11. Ringkasan
# ============================================================

print("=== Visualisasi selesai dibuat ===")
print(f"Hasil visualisasi disimpan di: {OUTPUT_DIR}")
print("")
print("File visualisasi:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.endswith(".png"):
        print(f"- {filename}")
