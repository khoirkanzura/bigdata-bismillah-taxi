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
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. Membuat Spark Session
# ============================================================

spark = SparkSession.builder \
    .appName("BISMILLAH_Taxi_Fare_Regression_HDFS_Spark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. Konfigurasi Path
# ============================================================

INPUT_PATH = "hdfs://10.83.19.247:9000/user/data/train_split_00.csv"
OUTPUT_BASE = "hdfs://10.83.19.247:9000/user/output/taxi_regression"

CLEANED_SAMPLE_OUTPUT = f"{OUTPUT_BASE}/cleaned_sample"
DATA_SUMMARY_OUTPUT = f"{OUTPUT_BASE}/data_summary"
EVALUATION_OUTPUT = f"{OUTPUT_BASE}/evaluation_result"
PREDICTION_OUTPUT = f"{OUTPUT_BASE}/prediction_sample"
FEATURE_IMPORTANCE_OUTPUT = f"{OUTPUT_BASE}/feature_importance"


# ============================================================
# 3. Load Dataset
# ============================================================

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

missing_columns = [c for c in required_columns if c not in df_raw.columns]

if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")

df = df_raw.select(*required_columns)


# ============================================================
# 5. Membersihkan Format Tanggal
# ============================================================

# Format asli contoh:
# 2009-06-15 17:26:21 UTC
# Bagian UTC dihapus agar lebih aman diparsing oleh Spark.

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

# Filter jarak tidak valid
df_clean = df_clean.filter(
    (col("trip_distance_km") > 0) &
    (col("trip_distance_km") <= 100)
)

total_after_distance_filter = df_clean.count()
print(f"Jumlah data setelah filter jarak: {total_after_distance_filter}")


# ============================================================
# 9. Feature Engineering: Fitur Waktu
# ============================================================

df_clean = df_clean.withColumn("pickup_hour", hour(col("pickup_timestamp")))
df_clean = df_clean.withColumn("pickup_dayofweek", dayofweek(col("pickup_timestamp")))
df_clean = df_clean.withColumn("pickup_month", month(col("pickup_timestamp")))
df_clean = df_clean.withColumn("pickup_year", year(col("pickup_timestamp")))

df_clean = df_clean.dropna()

total_final_clean = df_clean.count()
print(f"Jumlah data final setelah preprocessing: {total_final_clean}")


# ============================================================
# 10. Menyimpan Sampel Data Bersih
# ============================================================

df_clean.select(
    "fare_amount",
    "pickup_timestamp",
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
).limit(1000).coalesce(1).write.mode("overwrite").option("header", True).csv(CLEANED_SAMPLE_OUTPUT)


# ============================================================
# 11. Menyiapkan Fitur Model
# ============================================================

feature_columns = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_distance_km",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_year"
]

df_model = df_clean.withColumnRenamed("fare_amount", "label")

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

train_count = train_df.count()
test_count = test_df.count()

print(f"Jumlah data training: {train_count}")
print(f"Jumlah data testing: {test_count}")


# ============================================================
# 12. Evaluator Regresi
# ============================================================

rmse_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)


# ============================================================
# 13. Fungsi Training dan Evaluasi
# ============================================================

def train_and_evaluate(model_name, stages, train_data, test_data):
    print(f"\n=== Training Model: {model_name} ===")

    pipeline = Pipeline(stages=stages)
    fitted_model = pipeline.fit(train_data)
    predictions = fitted_model.transform(test_data)

    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)

    print(f"Model: {model_name}")
    print(f"RMSE : {rmse}")
    print(f"MAE  : {mae}")
    print(f"R2   : {r2}")

    return {
        "model_name": model_name,
        "pipeline_model": fitted_model,
        "predictions": predictions,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


# ============================================================
# 14. Pipeline Model
# ============================================================

# Linear Regression memakai StandardScaler
assembler_linear = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

linear_regression = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.1,
    elasticNetParam=0.0
)

# Tree-based models tidak wajib scaling
assembler_tree = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

decision_tree = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="label",
    maxDepth=8,
    seed=42
)

random_forest = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=30,
    maxDepth=10,
    seed=42
)

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
    maxDepth=6,
    seed=42
)


# ============================================================
# 15. Training Semua Model
# ============================================================

all_results = []

all_results.append(
    train_and_evaluate(
        "Linear Regression",
        [assembler_linear, scaler, linear_regression],
        train_df,
        test_df
    )
)

all_results.append(
    train_and_evaluate(
        "Decision Tree Regression",
        [assembler_tree, decision_tree],
        train_df,
        test_df
    )
)

all_results.append(
    train_and_evaluate(
        "Random Forest Regression",
        [assembler_tree, random_forest],
        train_df,
        test_df
    )
)

all_results.append(
    train_and_evaluate(
        "GBT Regression",
        [assembler_tree, gbt],
        train_df,
        test_df
    )
)


# ============================================================
# 16. Menentukan Model Terbaik
# ============================================================

best_result = min(all_results, key=lambda x: x["rmse"])

print("\n=== Model Terbaik ===")
print(f"Model terbaik: {best_result['model_name']}")
print(f"RMSE terbaik : {best_result['rmse']}")
print(f"MAE terbaik  : {best_result['mae']}")
print(f"R2 terbaik   : {best_result['r2']}")


# ============================================================
# 17. Simpan Hasil Evaluasi
# ============================================================

evaluation_rows = [
    (r["model_name"], r["rmse"], r["mae"], r["r2"])
    for r in all_results
]

evaluation_df = spark.createDataFrame(
    evaluation_rows,
    ["model", "rmse", "mae", "r2"]
)

evaluation_df.orderBy(col("rmse").asc()).show(truncate=False)

evaluation_df.coalesce(1).write.mode("overwrite").option("header", True).csv(EVALUATION_OUTPUT)


# ============================================================
# 18. Simpan Sampel Prediksi Model Terbaik
# ============================================================

best_predictions = best_result["predictions"]

best_predictions.select(
    "label",
    "prediction",
    "trip_distance_km",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_year"
).limit(1000).coalesce(1).write.mode("overwrite").option("header", True).csv(PREDICTION_OUTPUT)


# ============================================================
# 19. Simpan Model Terbaik
# ============================================================

best_model_path = f"{OUTPUT_BASE}/best_model_{best_result['model_name'].replace(' ', '_')}"
best_result["pipeline_model"].write().overwrite().save(best_model_path)

print(f"Model terbaik disimpan di: {best_model_path}")


# ============================================================
# 20. Feature Importance untuk Tree-Based Model
# ============================================================

if best_result["model_name"] in [
    "Decision Tree Regression",
    "Random Forest Regression",
    "GBT Regression"
]:
    try:
        final_model = best_result["pipeline_model"].stages[-1]
        importances = final_model.featureImportances.toArray().tolist()

        feature_importance_rows = [
            (feature_columns[i], float(importances[i]))
            for i in range(len(feature_columns))
        ]

        feature_importance_df = spark.createDataFrame(
            feature_importance_rows,
            ["feature", "importance"]
        )

        feature_importance_df.orderBy(col("importance").desc()).show(truncate=False)

        feature_importance_df.coalesce(1).write.mode("overwrite").option("header", True).csv(FEATURE_IMPORTANCE_OUTPUT)

    except Exception as e:
        print(f"Feature importance tidak bisa diambil: {e}")


# ============================================================
# 21. Simpan Ringkasan Data
# ============================================================

summary_rows = [
    ("Jumlah data awal", int(total_raw)),
    ("Jumlah data setelah dropna", int(total_after_dropna)),
    ("Jumlah data setelah filter dasar", int(total_after_basic_filter)),
    ("Jumlah data setelah filter jarak", int(total_after_distance_filter)),
    ("Jumlah data final setelah preprocessing", int(total_final_clean)),
    ("Jumlah data training", int(train_count)),
    ("Jumlah data testing", int(test_count))
]

summary_df = spark.createDataFrame(
    summary_rows,
    ["keterangan", "jumlah"]
)

summary_df.show(truncate=False)

summary_df.coalesce(1).write.mode("overwrite").option("header", True).csv(DATA_SUMMARY_OUTPUT)


# ============================================================
# 22. Selesai
# ============================================================

print("\n=== Eksperimen selesai ===")
print(f"Output disimpan di: {OUTPUT_BASE}")

spark.stop()
