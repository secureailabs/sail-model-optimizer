import pandas as pd

df = pd.read_csv("fatty_liver.csv")

# Convert feet and inches to just inches
list_value = []
for feet, inches in zip(df["Heightft"], df["Heightin"]):
    list_value.append((feet * 12) + inches)
df["height_inches"] = list_value

# Fix sex representation
df["Sex"] = df["Sex"].replace(1, 0)
df["Sex"] = df["Sex"].replace(2, 1)

# Swap NaNs for 0
list_value = ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
for ele in list_value:
    df[ele] = df[ele].fillna(0)

# Swap 2s for 1s
list_value = [
    "HepatitisBalone",
    "HepatitisBwithHepatitisD",
    "HepatitisC",
    "HepatitisE",
    "Autoimmunehepatitis",
    "Primarybiliarycholangitis",
    "Primarysclerosingcholangitis",
    "Ironoverload",
    "Alpha",
]
for ele in list_value:
    df[ele] = df[ele].replace(2, 1)
df.drop(columns=list_value)

# Create NAFLD Classification column
list_value = []
for ele in df["CAPdBm"].items():
    if ele[1] > 290:
        list_value.append(1)
    else:
        list_value.append(0)
df["NAFLD"] = list_value

# Remove upper ranges from cardio variable
df["Cardio"] = df["Cardio"].replace(3, 1)
df["Cardio"] = df["Cardio"].replace(4, 1)
df["Cardio"] = df["Cardio"].replace(2, 1)

# Remove upper ranges from strenght variable
df["Strength"] = df["Strength"].replace(2, 1)
df["Strength"] = df["Strength"].replace(3, 1)
df["Strength"] = df["Strength"].replace(4, 1)

# Split Vegetable consumption variable into upper and lower ranges
# print(df['Vegetables'].value_counts(sort=False))
# df['Vegetables'] = df['Vegetables'].replace(1, 0)
# df['Vegetables'] = df['Vegetables'].replace(2, 1)
# df['Vegetables'] = df['Vegetables'].replace(3, 1)
# df['Vegetables'] = df['Vegetables'].replace(4, 1)
# print(df['Vegetables'].value_counts(sort=False))


# drop irrelevant columns
df = df.drop(
    columns=[
        "Unnamed: 0",
        "ReportID",
        "Heightft",
        "Weight",
        "File",
        "havepatientinfo",
        "Race",
        "BMIcat",
        "CAPIQR",
        "CAPcat",
        "ISMIQR",
        "ISMIQRmed",
        "File",
        "ID",
        "ISMKPa",
        "TEcat",
        "Heightin",
        "AGEcat",
        "CAPdBm",
    ]
)

# Ensure float
df = df.astype(float)
df = df.dropna()


df_output = df["NAFLD"]
df_input = df.drop(columns="NAFLD")
df_output.to_csv("fatty_liver_output.csv", index=False)
df_input.to_csv("fatty_liver_input.csv", index=False)
