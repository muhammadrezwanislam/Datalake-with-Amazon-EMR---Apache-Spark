import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType,StructField,DoubleType,StringType, IntegerType,LongType
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import to_timestamp



config = configparser.ConfigParser()
config.read('dl.cfg')

#header of the dl.cfg file should have the same name as we in this variable [AWS]
os.environ['AWS_ACCESS_KEY_ID']=config["AWS"]['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config["AWS"]['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Establishes a Spark Session and returns it as an object
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Reads song-data from S3 bucket. Creates song_table and artist_table, and stores then in another S3 bucket in parquet format. 
    Arguments: 
        spark: SparkSession
        input_data: location of the input data 
        output_data: location of the output data 
        
    Returns:
        None 
    """
    #get filepath to song data file
    song_data = os.path.join(input_data,"song_data/*/*/*/*.json")
    
    #create song data schema
    song_schema = StructType([StructField('artist_id', StringType(), False),
        StructField('artist_latitude', StringType(), True),
        StructField('artist_longitude', StringType(), True),
        StructField('artist_location', StringType(), True),
        StructField('artist_name', StringType(), False),
        StructField('song_id', StringType(), False),
        StructField('title', StringType(), False),
        StructField('duration', DoubleType(), True),
        StructField('year', IntegerType(), True)
    ])
    
    # read song data file
    df = spark.read.json(song_data,schema=song_schema)

    # extract columns to create songs table based on unique song_id
    song_fields = ['song_id','title','artist_id','year','duration']
    songs_table = df.select(song_fields).dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs'), 'overwrite')
    print("(1/2) | songs completed")
    
    
    # extract columns to create artists table
    artist_fields = ['artist_id','artist_name','artist_location','artist_latitude','artist_longitude']
    #create artist table based on unique artist_id 
    artists_table = df.select(artist_fields) \
                    .withColumnRenamed('artist_name','artist') \
                    .withColumnRenamed('artist_location','location') \
                    .withColumnRenamed('artist_latitude','latitude') \
                    .withColumnRenamed('artist_longitude','longitude') \
                    .dropDuplicates(['artist_id'])
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists'), 'overwrite')
    print("(2/2) | artists completed")


def process_log_data(spark, input_data, output_data):
    """
     Reads log-data from base S3 bucket. Creates users_table,time_table, ang songplays_table and stores then in another S3 bucket in parquet format. 
    Arguments: 
        spark: SparkSession [Apache Spark Session]
        input_data: location of the input data 
        output_data: location of the output data 
        
    Returns:
        None 
    """
    
    # get filepath to log data file
    log_data =os.path.join(input_data,"log_data/*/*/*.json")
    #create log data schema
    log_schema = StructType([
        StructField('artist', StringType(), False),
        StructField('auth', StringType(), True),
        StructField('firstName', StringType(), True),
        StructField('gender', StringType(), True),
        StructField('itemInSession', LongType(), True),
        StructField('lastName', StringType(), True),
        StructField('length', DoubleType(), True),
        StructField('level', StringType(), True),
        StructField('location', StringType(), True),
        StructField('method', StringType(), True),
        StructField('page', StringType(), False),
        StructField('registration', DoubleType(), True),
        StructField('sessionId', LongType(), True),
        StructField('song', StringType(), False),
        StructField('status', LongType(), True),
        StructField('ts', LongType(), False),
        StructField('userAgent', StringType(), True),
        StructField('userId', StringType(), False),
    ])
    

    # read log data file
    df = spark.read.json(log_data, schema=log_schema)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table based on unique user_id
    user_fields = ['userId', 'firstName', 'lastName', 'gender', 'level']
    users_table = df.select(user_fields).dropDuplicates(['userId'])
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'), 'overwrite')
    print("(1/3) | users completed")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # extract columns to create time table
    time_table = time_table = df.select(
        col('timestamp').alias('start_time'),
        hour(col('timestamp')).alias('hour'),
        dayofmonth(col('timestamp')).alias('day'),
        weekofyear(col('timestamp')).alias('week'),
        month(col('timestamp')).alias('month'),
        year(col('timestamp')).alias('year')
    ).dropDuplicates(['start_time'])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'time'), 'overwrite')
    print("(2/3) | time completed")

    # read in artists data and song data to use for songplays table
    artists_table = spark.read.parquet(output_data +'artists')
    song_df = spark.read.parquet(output_data +'songs')
    songs = (
        song_df
        .join(artists_table, "artist_id", "full")
        .select("song_id", "title", "artist_id", "artist", "duration")
    )
                                   


    # extract columns from joined song and log datasets to create songplays table [check the left join]
    songplays_table = df.join(
        songs,
        [
            df.song == songs.title,
            df.artist == songs.artist,
            df.length == songs.duration
        ],
        "left"
    )

    # write songplays table to parquet files partitioned by year and month
    from pyspark.sql.functions import to_timestamp
    songplays_table = (
        songplays_table
        .join(time_table, [songplays_table.timestamp == time_table.start_time], "left")
        .select(
            "start_time",
            col("userId").alias("user_id"),
            "level",
            "song_id",
            "artist_id",
            col("sessionId").alias("session_id"),
            "location",
            col("userAgent").alias("user_agent"),
            "year",
            "month"
        )
        .withColumn("songplay_id", monotonically_increasing_id())
    )
    
    #write songplays table to parquet files partitioned by year and month 
    songplays_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'songplays'), 'overwrite')
    print("(3/3) | songplays completed")
    print("--- Completed Process Log_Data ---")
                                       

def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://apache-spark/output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
