# Evaluasi Model Regresi untuk Estimasi Tarif Taksi Berbasis Spark dan HDFS

Repository ini berisi implementasi eksperimen Big Data untuk melakukan estimasi tarif taksi menggunakan dataset NYC Taxi Fare. Proses pengolahan data dilakukan dengan memanfaatkan Hadoop Distributed File System (HDFS) dan Apache Spark untuk menangani data dalam skala besar. Selain itu, digunakan Spark ML Pipeline untuk membangun dan mengevaluasi model regresi yang mampu memprediksi tarif perjalanan berdasarkan berbagai fitur yang tersedia pada dataset.

## Anggota Kelompok

**Kelompok 5 - Sistem Informasi Bisnis Kelas 3D**

| Nama | NIM |
|------|-----|
| Adinda Ivanka Maysanda Putri | 2341760058 |
| Khoir Karol Nurzuraidah | 2341760048 |
| My Babby Findia Rudy Susanto | 2341760007 |
| Nadya Hapsari Putri | 2341760179 |
| Nindya Shafira Putri | 2341760059 |

## Teknologi yang Digunakan

- Apache Hadoop HDFS
- Apache Spark
- PySpark
- Spark ML Pipeline
- Python
- Pandas
- Matplotlib

## Tujuan Proyek

Proyek ini bertujuan untuk:

- Mengimplementasikan pemrosesan data berskala besar menggunakan Hadoop dan Spark.
- Melakukan pembersihan dan transformasi data NYC Taxi Fare.
- Membangun model regresi untuk memprediksi tarif taksi.
- Membandingkan performa beberapa algoritma regresi.
- Mengevaluasi hasil prediksi menggunakan metrik evaluasi yang sesuai.
- Menyajikan visualisasi hasil eksperimen untuk memudahkan analisis.

## Struktur Folder

```text
├── scripts/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
│
├── output/
│   ├── final/
│   └── visualizations/
│
├── screenshots/
│
└── README.md
```

## Dataset

Dataset yang digunakan adalah **NYC Taxi Fare Dataset**, yang berisi informasi perjalanan taksi di Kota New York, seperti:

- Jarak perjalanan
- Lokasi penjemputan dan pengantaran
- Waktu perjalanan
- Tarif perjalanan

Dataset ini digunakan sebagai dasar dalam proses pelatihan dan evaluasi model regresi.

## Output

Hasil yang dihasilkan dari proyek ini meliputi:

- Dataset yang telah dibersihkan dan ditransformasi.
- Model regresi yang telah dilatih menggunakan Spark ML.
- Hasil evaluasi model.
- Visualisasi performa model.
- Dokumentasi proses eksperimen.

## Dokumentasi

Folder `screenshots/` berisi dokumentasi berupa tangkapan layar proses implementasi, konfigurasi HDFS, eksekusi Spark, serta hasil evaluasi dan visualisasi yang diperoleh selama eksperimen berlangsung.
