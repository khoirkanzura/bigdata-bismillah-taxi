#!/bin/bash

echo "================================================"
echo "CEK STATUS CLUSTER HADOOP DAN SPARK"
echo "================================================"

echo ""
echo "1. Mengecek proses Java dengan jps..."
jps

echo ""
echo "2. Mengecek laporan DataNode HDFS..."
hdfs dfsadmin -report

echo ""
echo "3. Mengecek folder HDFS /user/data..."
hdfs dfs -ls /user/data

echo ""
echo "4. Mengecek file dataset di HDFS..."
hdfs dfs -ls -h /user/data/train_split_00.csv

echo ""
echo "5. Mengecek kondisi file dengan fsck..."
hdfs fsck /user/data/train_split_00.csv -files -blocks -locations

echo ""
echo "================================================"
echo "Pengecekan selesai."
echo "Pastikan laporan HDFS menunjukkan 3 DataNode aktif."
echo "================================================"
