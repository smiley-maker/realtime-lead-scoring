import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from src.data.prepare_data import define_data_schema

def create_spark_session(
        app_name : str = "SparkStreamingApp"
    ) -> tuple[SparkSession, pyspark.SparkContext]:
    """
    Create and return a Spark session.

    Args:
        app_name (str): Name of the Spark application.
    Returns:
        tuple: A tuple containing the SparkSession and SparkContext.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc

def load_model(spark: SparkSession, model_path: str) -> PipelineModel:
    """
    Load a pre-trained machine learning model.

    Args:
        spark (SparkSession): The Spark session.
        model_path (str): Path to the saved model.
    Returns:
        Pipeline: The loaded machine learning model.
    """
    model = PipelineModel.load(model_path)
    return model

def start_stream(spark: SparkSession, model: PipelineModel, input_path: str):
    """
    Start a Spark streaming job to process incoming data.

    Args:
        spark (SparkSession): The Spark session.
        model (PipelineModel): The pre-trained machine learning model.
        input_path (str): Path to the input data stream.
    """
    schema = define_data_schema()
    # Use the parquet format for the streaming source matching the input directory.
    sourceStream = spark.readStream.format("parquet").option("maxFilesPerTrigger", 1).schema(schema).load(input_path)
    query = model.transform(sourceStream).select("ProspectID", "ContactedAgent", "prediction", "probability")
    streamSink = query.writeStream.outputMode("append").format("console").trigger(processingTime='30 seconds').start()
    streamSink.awaitTermination()

if __name__ == "__main__":
    spark, sc = create_spark_session()
    print("Spark Session created with app name:", spark.sparkContext.appName)
    model_path = "models/lead_scoring_models/random_forest_base/"
    leadmodel = load_model(spark, model_path)
    print("Model loaded from:", model_path)
    input_path = "data/stream/user_events/"
    start_stream(spark, leadmodel, input_path)