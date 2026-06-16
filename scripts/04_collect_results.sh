#!/bin/bash

set -e

# ============================================================
# 04_collect_results.sh
# Mengumpulkan hasil preprocessing dan training Spark ke output/final
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/output"
FINAL_DIR="$OUTPUT_DIR/final"

echo "=== Mengumpulkan hasil eksperimen ==="
echo "Project directory: $PROJECT_DIR"
echo "Output directory : $OUTPUT_DIR"
echo "Final directory  : $FINAL_DIR"

mkdir -p "$FINAL_DIR"


# ============================================================
# Fungsi untuk menyalin file part CSV
# ============================================================

copy_csv_result() {
    SOURCE_DIR="$1"
    TARGET_FILE="$2"
    LABEL="$3"

    if ls "$SOURCE_DIR"/part-*.csv 1> /dev/null 2>&1; then
        cp "$SOURCE_DIR"/part-*.csv "$TARGET_FILE"
        echo "[OK] $LABEL disalin ke $TARGET_FILE"
    else
        echo "[SKIP] $LABEL tidak ditemukan di $SOURCE_DIR"
    fi
}


# ============================================================
# Mengumpulkan hasil ke output/final
# ============================================================

copy_csv_result "$OUTPUT_DIR/data_summary" "$FINAL_DIR/data_summary.csv" "Ringkasan data"
copy_csv_result "$OUTPUT_DIR/evaluation_result" "$FINAL_DIR/evaluation_result.csv" "Hasil evaluasi model"
copy_csv_result "$OUTPUT_DIR/prediction_sample" "$FINAL_DIR/prediction_sample.csv" "Sampel prediksi"
copy_csv_result "$OUTPUT_DIR/feature_importance" "$FINAL_DIR/feature_importance.csv" "Feature importance"


# ============================================================
# Menampilkan hasil akhir
# ============================================================

echo ""
echo "=== File final yang tersedia ==="
ls -lh "$FINAL_DIR"

echo ""
echo "=== Collect results selesai ==="
