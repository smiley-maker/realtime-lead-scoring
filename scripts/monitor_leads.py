import pyspark
from pyspark.sql import SparkSession

# Read incoming qualified leads data from a streaming source
def create_spark_session(
        app_name : str = "LeadMonitoringApp"
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

def query_stream(spark: SparkSession):
    """
    Query the streaming data for qualified leads.

    Args:
        spark (SparkSession): The Spark session.
        input_path (str): Path to the input data stream.
    """
    spark.sql("select * from leads").show(truncate=False)

if __name__ == "__main__":
    spark, sc = create_spark_session()
    print("Spark Session created with app name:", spark.sparkContext.appName)
    spark.sql("select * from leads").show(truncate=False)