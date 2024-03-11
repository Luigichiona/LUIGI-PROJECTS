from pyspark.shell import sqlContext
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import seaborn as sns


# Features explained of the dataset

#1. age

#2. sex:
#   1 = Male 0 = Female

#3. chest pain:
#   type Angina pain is often described as squeezing, pressure, heaviness, tightness or pain in the chest.
#   It may feel like a heavy weight lying on the chest.
#   !!Atypical pain is frequently defined as epigastric or back pain or pain that is described as burning, stabbing, or characteristic of indigestion.
#   !!Typical symptoms usually include chest, arm, or jaw pain described as dull, heavy, tight, or crushing
#   !!A chest pain is very likely nonanginal if its duration is over 30 minutes or less than 5 seconds,
#   it increases with inspiration, can be brought on with one movement of the trunk or arm, can be brought on by local fingers pressure, or bending forward, or it can be relieved immediately on lying down.
#   1 = Typical angina 2 = Atypical angina 3 = Non-anginal pain

#4. Resting blood pressure
#   120/80 trestbps - Resting Blood Pressure (in mm Hg on admission to the hospital)
#   However in some cases bp can go upto 200 too which means extremely critical.
#   A normal blood pressure level is less than 120/80 mmHg

#5. Serum cholestoral:
#   Serum cholestoral in mg/dl cholestrol can be more than 200 too..
#   so most probably I will ignore the outliers as it can be the case for someone so dropping or replacing outliers is not a good idea.
#   A total cholesterol level of less than 200 mg/dL (5.17 mmol/L) is normal.

#6. fasting blood sugar:
#   If fasting blood suger > 120 mg/dl:
#      0 = False 1 = True

#7. resting electrocardiographic results (values 0,1,2)
#   Results 0 = Hypertrophy 1 = Having ST-T wave abnormality 2 = Showing probable or definite left ventricular hypertrophy by Estes' criteria

#8. Exercise Induced Angina
#   0 = No 1 = Yes

#9. oldpeak = ST depression induced by exercise relative to rest

#10.the slope of the peak exercise ST segment slope
#   0 = Downsloping 1 = Flat 2 = Upsloping

#11. number of major vessels (0-3) colored by flourosopy

#12. thal:
#   0 = Null 1 = Fixed defect 2 = Normal 3 = Reversible defect

#13. target:
#   0= less chance of attack 1= more chance of heart attack


# Source dataset https://archive.ics.uci.edu/ml/datasets/Heart+Disease
filePath = "C:/Users/guusv/Documents/DSS/Interactive Data/heart.csv"

schema = StructType([
    StructField("age", IntegerType()),
    StructField("sex", IntegerType()),
    StructField("cp", IntegerType()),
    StructField("trestbps", IntegerType()),
    StructField("chol", IntegerType()),
    StructField("fbs", IntegerType()),
    StructField("restecg", IntegerType()),
    StructField("thalach", IntegerType()),
    StructField("exang", IntegerType()),
    StructField("oldpeak", DoubleType()),
    StructField("slope", IntegerType()),
    StructField("ca", IntegerType()),
    StructField("thal", IntegerType()),
    StructField("target", IntegerType())
])

#Load the dataset
dataset = (
    sqlContext
    .read
    .format("com.databricks.spark.csv")
    .schema(schema)
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load(filePath)
)

#First analytical task
heartattackSet = dataset.select(["chol"]).filter(f.col("target") == 1).rdd.flatMap(lambda x: x).collect()
noHeartattackSet = dataset.select(["chol"]).filter(f.col("target") == 0).rdd.flatMap(lambda x: x).collect()

avgChorestrolHA = sum(heartattackSet)/len(heartattackSet)
avgChorestrolNoHA = sum(noHeartattackSet)/len(noHeartattackSet)

print(f"\nAverage chorestrol level for a high chance of suffering a heart attack is {avgChorestrolHA}")
print(f"Meanwhile the average chorestrol level for a low chance of suffering a heart attack is {avgChorestrolNoHA}")
print(f"The difference between a low and high chance is {avgChorestrolHA-avgChorestrolNoHA}\n")

print("The chorestrol level of someone who has a high chance of suffering a heart attack is actually on average lower in comparison to someone who has a low chance of suffering a heart attack based on this dataset.")

#sns.set_theme()
#plt.hist([heartattackSet, noHeartattackSet], color=['r','b'], alpha=0.5)
#plt.title('RED = High chance, Blue = Low chance')
#plt.ylabel('Frequency')
#plt.xlabel('Cholestrol levels')
#plt.show()



#Second analytical task
bloodPressure = dataset.select(["trestbps"])

bloodPressure_HighChance = bloodPressure.filter(f.col("target") == 1).rdd.flatMap(lambda x: x).collect()
bloodPressure_lowChance = bloodPressure.filter(f.col("target") == 0).rdd.flatMap(lambda x: x).collect()

avgBloodPresHighHA = sum(bloodPressure_HighChance)/len(bloodPressure_HighChance)
avgBloodPresLowHA = sum(bloodPressure_lowChance)/len(bloodPressure_lowChance)

print(f"\nAverage blood pressure level for a high chance of suffering a heart attack is {avgBloodPresHighHA}")
print(f"Meanwhile the average blood pressure level for a low chance of suffering a heart attack is {avgBloodPresLowHA}")
print(f"The difference between a low and high chance is {avgBloodPresHighHA-avgBloodPresLowHA}\n")

#sns.set_theme()
#plt.hist([bloodPressure_HighChance, bloodPressure_lowChance], color=['r','b'], alpha=0.5)
#plt.title('Red = High chance, Blue = Low chance')
#plt.ylabel('Frequency')
#plt.xlabel('Resting blood pressure')
#plt.show()

fastingBloodSugar = dataset.select(["fbs"])

fastingBloodSugar_noPain = fastingBloodSugar.filter(f.col("cp") == 0).rdd.flatMap(lambda x: x).collect()
fastingBloodSugar_pain = fastingBloodSugar.filter(f.col("cp") > 0).rdd.flatMap(lambda x: x).collect()

sns.set_theme()
plt.hist([fastingBloodSugar_noPain, fastingBloodSugar_pain], color=['r','b'], alpha=0.5, bins=2)
plt.title('Red = No chest pain, Blue = Experience chest pain')
plt.ylabel('Frequency')
plt.xlabel('Fasting Blood Sugar')
plt.show()
