import json
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline

def create_spark_session():
    print("Initializing Spark Session...")
    return SparkSession.builder \
        .appName("CreditCardFraudDetection") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def load_data(spark, file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download it or run synthetic data generator.")
    
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    # Ensure Class is integer
    df = df.withColumn("Class", col("Class").cast("integer"))
    return df

def undersample_data(df):
    """
    Handles class imbalance by undersampling the majority class (Valid transactions, Class=0).
    """
    print("Handling Class Imbalance via Undersampling...")
    fraud_df = df.filter(col("Class") == 1)
    valid_df = df.filter(col("Class") == 0)
    
    fraud_count = fraud_df.count()
    valid_count = valid_df.count()
    
    print(f"Original Dataset -> Frauds: {fraud_count}, Valid: {valid_count}")
    
    if fraud_count == 0:
         raise ValueError("No fraud cases found in dataset!")

    # Calculate fraction to sample from valid_df
    fraction = fraud_count / valid_count
    
    # Sample valid dataframe to match fraud count (or slightly more)
    sampled_valid_df = valid_df.sample(withReplacement=False, fraction=fraction, seed=42)
    
    # Combine back
    balanced_df = fraud_df.unionAll(sampled_valid_df)
    
    new_fraud_ct = balanced_df.filter(col("Class")==1).count()
    new_valid_ct = balanced_df.filter(col("Class")==0).count()
    print(f"Balanced Dataset -> Frauds: {new_fraud_ct}, Valid: {new_valid_ct}")
    
    return balanced_df

def train_and_evaluate(df, model_name, classifier):
    print(f"\n--- Training {model_name} ---")
    
    # Assemble Features (V1..V28 + Time + Amount), dropping target 'Class'
    feature_cols = [c for c in df.columns if c != 'Class']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Create Pipeline
    pipeline = Pipeline(stages=[assembler, classifier])
    
    # Train-test split
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    print("Fitting model...")
    model = pipeline.fit(train_data)
    
    # Predict
    print("Making predictions...")
    predictions = model.transform(test_data)
    
    # Evaluators
    multi_evaluator_acc = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="accuracy")
    multi_evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="f1")
    multi_evaluator_prec = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="weightedPrecision")
    multi_evaluator_rec = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="weightedRecall")
    
    bin_evaluator_auc = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    # Calculate metrics
    accuracy = multi_evaluator_acc.evaluate(predictions)
    f1 = multi_evaluator_f1.evaluate(predictions)
    precision = multi_evaluator_prec.evaluate(predictions)
    recall = multi_evaluator_rec.evaluate(predictions)
    roc_auc = bin_evaluator_auc.evaluate(predictions)
    
    print(f"{model_name} Results: ACC={accuracy:.4f}, F1={f1:.4f}, PREC={precision:.4f}, REC={recall:.4f}, AUC={roc_auc:.4f}")
    
    # Calculate basic confusion matrix counts
    tp = predictions.filter("Class == 1 AND prediction == 1").count()
    tn = predictions.filter("Class == 0 AND prediction == 0").count()
    fp = predictions.filter("Class == 0 AND prediction == 1").count()
    fn = predictions.filter("Class == 1 AND prediction == 0").count()
    
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }
    }
    return model, results

def main():
    # Attempt to use real kaggle data, fallback to synthetic
    data_paths = ["data/creditcard.csv", "data/synthetic_creditcard.csv"]
    target_path = None
    
    for path in data_paths:
        if os.path.exists(path):
            target_path = path
            break
            
    if not target_path:
        print("No datasets found! Please run `python src/generate_synthetic.py` first.")
        return

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    # Load and balance
    df = load_data(spark, target_path)
    balanced_df = undersample_data(df)
    
    # Define models to train
    lr = LogisticRegression(labelCol="Class", featuresCol="features", maxIter=10)
    rf = RandomForestClassifier(labelCol="Class", featuresCol="features", numTrees=20)
    
    # Store metrics
    all_metrics = []
    
    # Train Logistic Regression
    lr_model, lr_metrics = train_and_evaluate(balanced_df, "Logistic Regression", lr)
    all_metrics.append(lr_metrics)
    
    # Train Random Forest
    rf_model, rf_metrics = train_and_evaluate(balanced_df, "Random Forest", rf)
    all_metrics.append(rf_metrics)
    
    # Ensure output dir exists
    os.makedirs("output", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    with open("output/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    print("\nSaving Random Forest model for Real-Time Streaming...")
    model_path = "saved_models/rf_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    rf_model.save(model_path)
        
    print("\nPySpark Pipeline Completed! Metrics saved to output/metrics.json")
    print(f"Model saved to {model_path} for Streaming Inference.")
    
    spark.stop()

if __name__ == "__main__":
    main()
