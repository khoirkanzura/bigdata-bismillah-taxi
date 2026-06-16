#!/bin/bash

echo "================================================"
echo "PERSIAPAN DATASET DI HDFS"
echo "================================================"

HDFS_FILE="/user/data/train_split_00.csv"
OUTPUT_DIR="/user/output/taxi_regression"

echo ""
echo "1. Membuat folder output HDFS..."
hdfs dfs -mkdir -p "$OUTPUT_DIR"

echo ""
echo "2. Mengecek apakah dataset sudah ada di HDFS..."
if hdfs dfs -test -e "$HDFS_FILE"; then
    echo "Dataset ditemukan di HDFS:"
    hdfs dfs -ls -h "$HDFS_FILE"
else
    echo "ERROR: Dataset tidak ditemukan di HDFS."
    echo "Pastikan file tersedia di:"
    echo "$HDFS_FILE"
    exit 1
fi

echo ""
echo "3. Mengatur replication factor menjadi 3..."
hdfs dfs -setrep -w 3 "$HDFS_FILE"

echo ""
echo "4. Mengecek kondisi file setelah set replication..."
hdfs fsck "$HDFS_FILE" -files -blocks -locations

echo ""
echo "================================================"
echo "Persiapan HDFS selesai."
echo "================================================"
