from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    regexp_replace,
    to_timestamp,
    hour,
    dayofweek,
    month,
    year,
    radians,
    sin,
    cos,
    asin,
    sqrt,
    lit
)


# ============================================================
# 1. Membuat Spark Session
# ============================================================

spark = SparkSession.builder \
    .appName("BISMILLAH_Preprocess_Taxi_To_Parquet") \
    .config("spark.sql.shuffle.partitions", "12") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. Path HDFS
# ============================================================

INPUT_PATH = "hdfs://10.83.19.247:9000/user/data/train_split_00.csv"

OUTPUT_BASE = "hdfs://10.83.19.247:9000/user/output/taxi_regression"
CLEAN_PARQUET_OUTPUT = f"{OUTPUT_BASE}/clean_parquet"
DATA_SUMMARY_OUTPUT = f"{OUTPUT_BASE}/data_summary"
CLEANED_SAMPLE_OUTPUT = f"{OUTPUT_BASE}/cleaned_sample"


# ============================================================
# 3. Membaca Dataset dari HDFS
# ============================================================

print("\n=== Membaca dataset dari HDFS ===")
print(f"Input path: {INPUT_PATH}")

df_raw = spark.read.csv(
    INPUT_PATH,
    header=True,
    inferSchema=True
)

print("\n=== Schema Dataset Awal ===")
df_raw.printSchema()

total_raw = df_raw.count()
print(f"\nJumlah data awal: {total_raw}")


# ============================================================
# 4. Validasi Kolom Dataset
# ============================================================

required_columns = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]

missing_columns = [column for column in required_columns if column not in df_raw.columns]

if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")

df = df_raw.select(*required_columns)


# ============================================================
# 5. Membersihkan Format Tanggal
# ============================================================

# Format tanggal asli pada dataset:
# 2009-06-15 17:26:21 UTC
# Bagian " UTC" dihapus agar lebih aman diproses oleh Spark.

df = df.withColumn(
    "pickup_datetime_clean",
    regexp_replace(col("pickup_datetime"), " UTC", "")
)

df = df.withColumn(
    "pickup_timestamp",
    to_timestamp(col("pickup_datetime_clean"), "yyyy-MM-dd HH:mm:ss")
)


# ============================================================
# 6. Menghapus Missing Value
# ============================================================

df_not_null = df.dropna(subset=[
    "fare_amount",
    "pickup_timestamp",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
])

total_after_dropna = df_not_null.count()
print(f"Jumlah data setelah dropna: {total_after_dropna}")


# ============================================================
# 7. Filter Data Tidak Valid
# ============================================================

# Filter:
# 1. Tarif harus lebih dari 0 dan tidak terlalu ekstrem.
# 2. Jumlah penumpang harus 1 sampai 6.
# 3. Koordinat dibatasi pada area sekitar New York.

df_clean = df_not_null.filter(
    (col("fare_amount") > 0) &
    (col("fare_amount") <= 300) &
    (col("passenger_count") > 0) &
    (col("passenger_count") <= 6) &
    (col("pickup_longitude").between(-75, -72)) &
    (col("dropoff_longitude").between(-75, -72)) &
    (col("pickup_latitude").between(40, 42)) &
    (col("dropoff_latitude").between(40, 42))
)

total_after_basic_filter = df_clean.count()
print(f"Jumlah data setelah filter dasar: {total_after_basic_filter}")


# ============================================================
# 8. Feature Engineering: Menghitung Jarak Haversine
# ============================================================

# Rumus Haversine digunakan untuk menghitung jarak antara
# titik pickup dan dropoff berdasarkan latitude dan longitude.

earth_radius_km = 6371.0

lat1 = radians(col("pickup_latitude"))
lon1 = radians(col("pickup_longitude"))
lat2 = radians(col("dropoff_latitude"))
lon2 = radians(col("dropoff_longitude"))

dlat = lat2 - lat1
dlon = lon2 - lon1

a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
c = 2 * asin(sqrt(a))

df_clean = df_clean.withColumn(
    "trip_distance_km",
    lit(earth_radius_km) * c
)

# Filter jarak yang tidak valid.
# Jarak 0 berarti pickup dan dropoff sama atau data bermasalah.
# Jarak lebih dari 100 km dianggap outlier untuk perjalanan taksi NYC.

df_clean = df_clean.filter(
    (col("trip_distance_km") > 0) &
    (col("trip_distance_km") <= 100)
)

total_after_distance_filter = df_clean.count()
print(f"Jumlah data setelah filter jarak: {total_after_distance_filter}")


# ============================================================
# 9. Feature Engineering: Ekstraksi Fitur Waktu
# ============================================================

df_clean = df_clean.withColumn("pickup_hour", hour(col("pickup_timestamp")))
df_clean = df_clean.withColumn("pickup_dayofweek", dayofweek(col("pickup_timestamp")))
df_clean = df_clean.withColumn("pickup_month", month(col("pickup_timestamp")))
df_clean = df_clean.withColumn("pickup_year", year(col("pickup_timestamp")))

df_clean = df_clean.dropna()

total_final_clean = df_clean.count()
print(f"Jumlah data final setelah preprocessing: {total_final_clean}")


# ============================================================
# 10. Memilih Kolom Final untuk Modeling
# ============================================================

final_columns = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "trip_distance_km",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_year"
]

df_final = df_clean.select(*final_columns)


print("\n=== Schema Data Final ===")
df_final.printSchema()


# ============================================================
# 11. Menyimpan Data Bersih ke Parquet
# ============================================================

print("\n=== Menyimpan data bersih ke Parquet ===")
print(f"Output path: {CLEAN_PARQUET_OUTPUT}")

df_final.write.mode("overwrite").parquet(CLEAN_PARQUET_OUTPUT)


# ============================================================
# 12. Menyimpan Sampel Data Bersih untuk Visualisasi
# ============================================================

print("\n=== Menyimpan sampel data bersih ===")
print(f"Output path: {CLEANED_SAMPLE_OUTPUT}")

df_final.limit(1000) \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(CLEANED_SAMPLE_OUTPUT)


# ============================================================
# 13. Menyimpan Ringkasan Jumlah Data
# ============================================================

summary_rows = [
    ("Jumlah data awal", int(total_raw)),
    ("Jumlah data setelah dropna", int(total_after_dropna)),
    ("Jumlah data setelah filter dasar", int(total_after_basic_filter)),
    ("Jumlah data setelah filter jarak", int(total_after_distance_filter)),
    ("Jumlah data final setelah preprocessing", int(total_final_clean))
]

summary_df = spark.createDataFrame(
    summary_rows,
    ["keterangan", "jumlah"]
)

print("\n=== Ringkasan Data ===")
summary_df.show(truncate=False)

summary_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(DATA_SUMMARY_OUTPUT)


# ============================================================
# 14. Selesai
# ============================================================

print("\n=== Preprocessing selesai ===")
print(f"Data bersih disimpan di: {CLEAN_PARQUET_OUTPUT}")
print(f"Ringkasan data disimpan di: {DATA_SUMMARY_OUTPUT}")
print(f"Sampel data bersih disimpan di: {CLEANED_SAMPLE_OUTPUT}")

spark.stop()
