from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count, avg, broadcast
from pandasgui import show
import os, glob, shutil

spark = SparkSession.builder.appName("TestSpark").getOrCreate()
print("Spark initialisé ✅")

df = spark.read.csv("data/personality_datasert.csv", header=True, inferSchema=True)      # Je définie et charge le dataset des personnalité Extravertie/Introvertie
df = df.withColumnRenamed("Stage_fear", "Vocal_fear")   # Modification du nom de colonne Stage_fear => Vocal_fear (plus parlant à mes yeux)
df.show(5)
df_clean = df.dropna()  # On enlève les valeur vide 

# Transformation des strings en valeur (binaire)

df = df.withColumn(
    "Vocal_fear", when(col("Vocal_fear") == "Yes", 1).otherwise(0)
).withColumn(
    "Drained_after_socializing", when(col("Drained_after_socializing") == "Yes", 1).otherwise(0)
)
df = df.withColumn(
    "Personality", when(col("Personality") == "Extrovert", 1).otherwise(0)
)

# Définie toutes les colonnes comme étant des doubles

numerical_cols = [
    "Time_spent_Alone", "Social_event_attendance", "Going_outside",
    "Friends_circle_size", "Post_frequency"
]
for col_name in numerical_cols:
    df_clean = df_clean.withColumn(col_name, col(col_name).cast("double"))  # On boucle toutes les colonnes

# Conversion en pandas
pdf = df.toPandas()  
# Lance la fenêtre interactive
show(pdf)

pdf.to_csv("data/personality_cleaned.csv", index=False)  
print("📄 CSV créé directement dans data/personality_cleaned.csv")