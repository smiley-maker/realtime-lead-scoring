import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, LongType, DoubleType, TimestampType, StructType, StructField

def create_spark_session(
        app_name : str = "DataPreparationApp"
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

def define_data_schema() -> StructType:
    """
    Define and return the schema for the dataset.

    Returns:
        StructType: The schema of the dataset.
    """
    zillowSchema = StructType([
        StructField('ProspectID', StringType(), True),
        StructField('LeadNumber', LongType(), True),
        StructField('LeadCaptureChannel', StringType(), True),
        StructField('ReferralSource', StringType(), True),
        StructField('OptOutEmail', StringType(), True),
        StructField('OptOutCall', StringType(), True),
        StructField('ContactedAgent', LongType(), True),
        StructField('TotalVisits', LongType(), True),
        StructField('TotalBrowsingTime', LongType(), True),
        StructField('AvgListingsViewedPerSession', DoubleType(), True),
        StructField('LastAction', StringType(), True),
        StructField('Country', StringType(), True),
        StructField('Specialization', StringType(), True),
        StructField('How did you hear about Zillow', StringType(), True),
        StructField('What is your current occupation', StringType(), True),
        StructField('What matters most to you in choosing this house', StringType(), True),
        StructField('Search', StringType(), True),
        StructField('Magazine', StringType(), True),
        StructField('Newspaper Article', StringType(), True),
        StructField('Zillow Forums', StringType(), True),
        StructField('Newspaper', StringType(), True),
        StructField('Digital Advertisement', StringType(), True),
        StructField('Through Recommendations', StringType(), True),
        StructField('Receive More Updates About Our Houses', StringType(), True),
        StructField('LeadStatusTag', StringType(), True),
        StructField('LeadQuality', StringType(), True),
        StructField('Update me on Zillow Content', StringType(), True),
        StructField('Get updates on DM Content', StringType(), True),
        StructField('LeadProfile', StringType(), True),
        StructField('City', StringType(), True),
        StructField('Asymmetric_Activity_Index', StringType(), True),
        StructField('Asymmetric_Profile_Index', StringType(), True),
        StructField('Asymmetric_Activity_Score', LongType(), True),
        StructField('Asymmetric_Profile_Score', LongType(), True),
        StructField('I agree to pay the amount through cheque', StringType(), True),
        StructField('a free copy of House Buying 101', StringType(), True),
        StructField('FinalEngagementAction', StringType(), True)
    ])

    return zillowSchema

def load_data(
        spark: SparkSession,
        file_path: str,
        schema: StructType
    ) -> tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """
    Load data from a CSV file into a Spark DataFrame using the provided schema.

    Args:
        spark (SparkSession): The Spark session.
        file_path (str): The path to the CSV file.
        schema (StructType): The schema to apply to the DataFrame.

    Returns:
        DataFrame: The loaded Spark DataFrame.
    """
    df = spark.read.option("dropMalformed", True) \
                    .option("ignoreLeadingWhiteSpace", True) \
                    .option("ignoreTrailingWhiteSpace", True) \
                    .csv(file_path, header=True, schema=schema)

    
    df = df.select([
        "ContactedAgent", "TotalVisits", "TotalBrowsingTime", "AvgListingsViewedPerSession",
        "LeadCaptureChannel", "ReferralSource", "LastAction", "FinalEngagementAction",
        "City", "Country", "LeadStatusTag", "ProspectID", "LeadNumber"
    ])

    df = df.dropna(subset=['ContactedAgent'])  # Can't train without labels

    train_df, stream_df = df.randomSplit([0.7, 0.3], seed=42)

    return train_df, stream_df

# Add at the end to actually save the data
if __name__ == "__main__":
    spark, sc = create_spark_session()
    schema = define_data_schema()
    train_df, stream_df = load_data(spark, "data/raw/Lead Scoring.csv", schema)
    
    print("Training Data Sample:")
    train_df.show(5)
    print(f"Training data count: {train_df.count()}")
    
    print("\nStreaming Data Sample:")
    stream_df.show(5)
    print(f"Streaming data count: {stream_df.count()}")
    
    # Save training data
    print("\nSaving training data...")
    train_df.write.mode('overwrite').parquet('data/processed/training_data.parquet')
    
    # Save streaming data (repartitioned for simulation)
    print("Saving streaming data (50 partitions for streaming simulation)...")
    stream_df.repartition(100).write.mode('overwrite').parquet('data/stream/user_events/')
    
    print("\n✅ Data preparation complete!")
    print("  - Training: data/processed/training_data.parquet")
    print("  - Streaming: data/streaming/user_events/")
    
    spark.stop()