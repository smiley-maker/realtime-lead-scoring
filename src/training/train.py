import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    Bucketizer,
    VectorAssembler,
    StandardScaler,
    OneHotEncoder,
    Imputer,
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import json


def create_spark_session(
    app_name: str = "DataPreparationApp",
) -> tuple[SparkSession, pyspark.SparkContext]:
    """
    Create and return a Spark session.

    Args:
        app_name (str): Name of the Spark application.
    Returns:
        tuple: A tuple containing the SparkSession and SparkContext.
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def load_data(spark: SparkSession, file_path: str) -> pyspark.sql.DataFrame:
    """
    Load training data from Parquet file.

    Args:
        spark (SparkSession): The Spark session.
        file_path (str): Path to the Parquet file.

    Returns:
        DataFrame: The training DataFrame.
    """
    df = spark.read.parquet(file_path)  # Schema embedded in Parquet
    return df


def create_pipeline() -> Pipeline:
    """
    Create and return a machine learning pipeline.

    Returns:
        Pipeline: The machine learning pipeline.
    """
    pipeline = Pipeline(
        stages=[
            Imputer(
                inputCols=[
                    "TotalVisits",
                    "TotalBrowsingTime",
                    "AvgListingsViewedPerSession",
                ],
                outputCols=[
                    "TotalVisits",
                    "TotalBrowsingTime",
                    "AvgListingsViewedPerSession",
                ],
            ),
            Bucketizer(
                inputCol="TotalVisits",
                outputCol="TotalVisits_bucket",
                splits=[0, 10, 20, 50, 75, 100, 150, 200, 300, float("Inf")],
            ),
            StringIndexer(
                inputCol="LeadCaptureChannel",
                outputCol="LeadCaptureChannel_index",
                handleInvalid="keep",
            ),
            OneHotEncoder(
                inputCol="LeadCaptureChannel_index", outputCol="LeadCaptureChannel_ohe"
            ),
            StringIndexer(
                inputCol="ReferralSource",
                outputCol="ReferralSource_index",
                handleInvalid="keep",
            ),
            OneHotEncoder(
                inputCol="ReferralSource_index", outputCol="ReferralSource_ohe"
            ),
            StringIndexer(
                inputCol="LastAction",
                outputCol="LastAction_index",
                handleInvalid="keep",
            ),
            OneHotEncoder(inputCol="LastAction_index", outputCol="LastAction_ohe"),
            StringIndexer(
                inputCol="FinalEngagementAction",
                outputCol="FinalEngagementAction_index",
                handleInvalid="keep",
            ),
            OneHotEncoder(
                inputCol="FinalEngagementAction_index",
                outputCol="FinalEngagementAction_ohe",
            ),
            StringIndexer(
                inputCol="City", outputCol="City_index", handleInvalid="keep"
            ),
            OneHotEncoder(inputCol="City_index", outputCol="City_ohe"),
            StringIndexer(
                inputCol="Country", outputCol="Country_index", handleInvalid="keep"
            ),
            OneHotEncoder(inputCol="Country_index", outputCol="Country_ohe"),
            StringIndexer(
                inputCol="LeadStatusTag",
                outputCol="LeadStatusTag_index",
                handleInvalid="keep",
            ),
            OneHotEncoder(
                inputCol="LeadStatusTag_index", outputCol="LeadStatusTag_ohe"
            ),
            VectorAssembler(
                inputCols=[
                    "TotalVisits_bucket",
                    "TotalBrowsingTime",
                    "AvgListingsViewedPerSession",
                    "LeadCaptureChannel_ohe",
                    "ReferralSource_ohe",
                    "LastAction_ohe",
                    "City_ohe",
                    "Country_ohe",
                    "LeadStatusTag_ohe",
                ],
                outputCol="features",
            ),
            RandomForestClassifier(featuresCol="features", labelCol="ContactedAgent"),
        ]
    )
    return pipeline


def save_model(model: pyspark.ml.PipelineModel, path: str) -> None:
    """
    Save the trained model to the specified path.

    Args:
        model (PipelineModel): The trained machine learning model.
        path (str): The path to save the model.
    """
    model.write().overwrite().save(path)


def train_model(
    train_df: pyspark.sql.DataFrame, pipeline: Pipeline
) -> pyspark.ml.PipelineModel:
    """
    Train the machine learning model using the provided training DataFrame and pipeline.

    Args:
        train_df (DataFrame): The training DataFrame.
        pipeline (Pipeline): The machine learning pipeline.

    Returns:
        PipelineModel: The trained machine learning model.
    """
    model = pipeline.fit(train_df)
    return model


def evaluate_model(
    model: pyspark.ml.PipelineModel, test_df: pyspark.sql.DataFrame
) -> dict:
    """
    Evaluate the trained model on test data.

    Args:
        model: The trained pipeline model.
        test_df: Test DataFrame.

    Returns:
        dict: Evaluation metrics.
    """
    predictions = model.transform(test_df)

    # Binary classification metrics
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="ContactedAgent", metricName="areaUnderROC"
    )
    auc = binary_evaluator.evaluate(predictions)

    # Accuracy
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="ContactedAgent", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = multi_evaluator.evaluate(predictions)

    # Precision/Recall
    multi_evaluator.setMetricName("weightedPrecision")
    precision = multi_evaluator.evaluate(predictions)

    multi_evaluator.setMetricName("weightedRecall")
    recall = multi_evaluator.evaluate(predictions)

    metrics = {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    return metrics


def save_metrics(
    metrics: dict, path: str = "models/metrics/training_metrics.json"
) -> None:
    """Save model metrics to JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Error saving metrics: {e}")

    print(f"\nMetrics saved to {path}")


if __name__ == "__main__":
    spark, sc = create_spark_session()

    # Load data
    print("Loading training data...")
    full_df = load_data(spark, "data/processed/training_data.parquet")

    # Train/test split
    train_df, test_df = full_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training samples: {train_df.count()}")
    print(f"Test samples: {test_df.count()}")

    # Create and train pipeline
    print("\nTraining model...")
    pipeline = create_pipeline()
    model = train_model(train_df, pipeline)

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_df)

    print("\n" + "=" * 50)
    print("Model Performance Metrics:")
    print("=" * 50)
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 50)

    # Save model and metrics
    print("\nSaving model...")
    save_model(model, "models/lead_scoring_model")
    save_metrics(metrics)

    print("\n✅ Model training complete!")
    print("  - Model: models/lead_scoring_models/random_forest_base/")
    print("  - Metrics: models/metrics/training_metrics.json")

    spark.stop()
