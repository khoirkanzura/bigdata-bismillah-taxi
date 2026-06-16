from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.storagelevel import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. Konfigurasi Utama
# ============================================================

INPUT_PARQUET = "hdfs://10.83.19.247:9000/user/output/taxi_regression/clean_parquet"

OUTPUT_BASE = "hdfs://10.83.19.247:9000/user/output/taxi_regression"
EVALUATION_OUTPUT = f"{OUTPUT_BASE}/evaluation_result"
PREDICTION_OUTPUT = f"{OUTPUT_BASE}/prediction_sample"
FEATURE_IMPORTANCE_OUTPUT = f"{OUTPUT_BASE}/feature_importance"
MODEL_OUTPUT = f"{OUTPUT_BASE}/best_model"

SAVE_BEST_MODEL = False


# ============================================================
# 2. Membuat Spark Session
# ============================================================

spark = SparkSession.builder \
    .appName("BISMILLAH_Train_Taxi_Regression_Safe") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 3. Membaca Data Bersih dari HDFS Parquet
# ============================================================

print("\n=== Membaca data bersih dari Parquet HDFS ===")
print(f"Input path: {INPUT_PARQUET}")

df = spark.read.parquet(INPUT_PARQUET)

print("\n=== Schema Data Bersih ===")
df.printSchema()

print("\nJumlah data bersih berdasarkan hasil preprocessing: 984372")


# ============================================================
# 4. Menyiapkan Dataset untuk Modeling
# ============================================================

df_model = df.withColumnRenamed("fare_amount", "label")

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

required_columns = ["label"] + feature_columns
missing_columns = [column for column in required_columns if column not in df_model.columns]

if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan dalam data bersih: {missing_columns}")

df_model = df_model.select(*required_columns)


# ============================================================
# 5. Split Data Training dan Testing
# ============================================================

print("\n=== Membagi data training dan testing ===")

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

train_df = train_df.persist(StorageLevel.DISK_ONLY)
test_df = test_df.persist(StorageLevel.DISK_ONLY)

train_count = train_df.count()
test_count = test_df.count()

print(f"Jumlah data training: {train_count}")
print(f"Jumlah data testing: {test_count}")


# ============================================================
# 6. Evaluator Regresi
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
# 7. Fungsi Training dan Evaluasi Model
# ============================================================

def evaluate_model(model_name, stages):
    print(f"\n=== Training Model: {model_name} ===")

    try:
        pipeline = Pipeline(stages=stages)
        fitted_model = pipeline.fit(train_df)

        predictions = fitted_model.transform(test_df).persist(StorageLevel.DISK_ONLY)

        rmse = rmse_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)
        r2 = r2_evaluator.evaluate(predictions)

        print(f"Model: {model_name}")
        print(f"RMSE : {rmse}")
        print(f"MAE  : {mae}")
        print(f"R2   : {r2}")

        return {
            "success": True,
            "model_name": model_name,
            "pipeline_model": fitted_model,
            "predictions": predictions,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    except Exception as error:
        print(f"\nModel {model_name} gagal dijalankan.")
        print(f"Error: {error}")

        return {
            "success": False,
            "model_name": model_name,
            "pipeline_model": None,
            "predictions": None,
            "rmse": None,
            "mae": None,
            "r2": None
        }


# ============================================================
# 8. Pipeline Linear Regression
# ============================================================

assembler_linear = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=False
)

linear_regression = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.1,
    elasticNetParam=0.0
)


# ============================================================
# 9. Pipeline Tree-Based Model
# ============================================================

assembler_tree = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

decision_tree = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="label",
    maxDepth=6,
    seed=42
)

random_forest = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=5,
    maxDepth=5,
    seed=42
)


# ============================================================
# 10. Training Model
# ============================================================

results = []

results.append(
    evaluate_model(
        "Linear Regression",
        [assembler_linear, scaler, linear_regression]
    )
)

results.append(
    evaluate_model(
        "Decision Tree Regression",
        [assembler_tree, decision_tree]
    )
)

results.append(
    evaluate_model(
        "Random Forest Regression",
        [assembler_tree, random_forest]
    )
)


# ============================================================
# 11. Mengambil Model yang Berhasil
# ============================================================

successful_results = [
    result for result in results
    if result["success"]
]

if not successful_results:
    raise RuntimeError("Semua model gagal dijalankan. Tidak ada hasil evaluasi yang bisa disimpan.")

best_result = min(successful_results, key=lambda result: result["rmse"])

print("\n=== Model Terbaik Berdasarkan RMSE ===")
print(f"Model terbaik: {best_result['model_name']}")
print(f"RMSE terbaik : {best_result['rmse']}")
print(f"MAE terbaik  : {best_result['mae']}")
print(f"R2 terbaik   : {best_result['r2']}")


# ============================================================
# 12. Menyimpan Hasil Evaluasi Model
# ============================================================

evaluation_rows = [
    (
        result["model_name"],
        result["rmse"],
        result["mae"],
        result["r2"]
    )
    for result in successful_results
]

evaluation_df = spark.createDataFrame(
    evaluation_rows,
    ["model", "rmse", "mae", "r2"]
)

print("\n=== Hasil Evaluasi Semua Model yang Berhasil ===")
evaluation_df.orderBy(col("rmse").asc()).show(truncate=False)

print(f"\nMenyimpan hasil evaluasi ke: {EVALUATION_OUTPUT}")

evaluation_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(EVALUATION_OUTPUT)


# ============================================================
# 13. Menyimpan Sampel Prediksi Model Terbaik
# ============================================================

best_predictions = best_result["predictions"]

print(f"\nMenyimpan sampel prediksi ke: {PREDICTION_OUTPUT}")

best_predictions.select(
    "label",
    "prediction",
    "trip_distance_km",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_year"
).limit(1000) \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(PREDICTION_OUTPUT)


# ============================================================
# 14. Menyimpan Feature Importance Jika Model Terbaik Tree-Based
# ============================================================

if best_result["model_name"] in [
    "Decision Tree Regression",
    "Random Forest Regression"
]:
    try:
        final_model = best_result["pipeline_model"].stages[-1]
        importances = final_model.featureImportances.toArray().tolist()

        feature_importance_rows = [
            (
                feature_columns[index],
                float(importances[index])
            )
            for index in range(len(feature_columns))
        ]

        feature_importance_df = spark.createDataFrame(
            feature_importance_rows,
            ["feature", "importance"]
        )

        print("\n=== Feature Importance Model Terbaik ===")
        feature_importance_df.orderBy(col("importance").desc()).show(truncate=False)

        print(f"\nMenyimpan feature importance ke: {FEATURE_IMPORTANCE_OUTPUT}")

        feature_importance_df.coalesce(1) \
            .write \
            .mode("overwrite") \
            .option("header", True) \
            .csv(FEATURE_IMPORTANCE_OUTPUT)

    except Exception as error:
        print(f"Feature importance tidak bisa diambil: {error}")

else:
    print("\nModel terbaik bukan tree-based model, feature importance tidak dibuat.")


# ============================================================
# 15. Menyimpan Model Terbaik Jika Diperlukan
# ============================================================

if SAVE_BEST_MODEL:
    best_model_name_safe = best_result["model_name"].replace(" ", "_")
    best_model_path = f"{MODEL_OUTPUT}_{best_model_name_safe}"

    print(f"\nMenyimpan model terbaik ke: {best_model_path}")

    best_result["pipeline_model"] \
        .write \
        .overwrite() \
        .save(best_model_path)

    print(f"Model terbaik disimpan di: {best_model_path}")
else:
    print("\nPenyimpanan model terbaik dilewati karena SAVE_BEST_MODEL = False.")


# ============================================================
# 16. Membersihkan Persist
# ============================================================

for result in successful_results:
    if result["predictions"] is not None:
        result["predictions"].unpersist()

train_df.unpersist()
test_df.unpersist()


# ============================================================
# 17. Selesai
# ============================================================

print("\n=== Training dan Evaluasi Model Selesai ===")
print(f"Hasil evaluasi disimpan di: {EVALUATION_OUTPUT}")
print(f"Sampel prediksi disimpan di: {PREDICTION_OUTPUT}")

spark.stop()
