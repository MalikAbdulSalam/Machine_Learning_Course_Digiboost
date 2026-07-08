import pandas as pd

# Load the CSV file
df = pd.read_csv("data.csv")

# Display the first 10 rows
print(df.head(1))

print("###############################  Display last 10 rows ################")
# Display last 10 rows
print(df.tail(10))



print("###############################  shape of data (culums, row) ################")
# Display the shape of the dataset
print(df.shape)


print("###############################  show all columnes ################")
# Display all column names
print(df.columns)

print("###############################  show all columnes datatypes ################")
# Display data types of all columns
data_type_of_all_columns = df.dtypes
print(data_type_of_all_columns)

print("###############################  statistics of dataset ################")

# Generate descriptive statistics
print(df.describe())

print("###############################  check duplicate values  ################")
# Check for duplicate rows
print(df.duplicated())


# Count the total number of duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

print("###############################  connver column into dateformat  ################")

# Convert the Date column to datetime format
#df["Date"] = pd.to_datetime(df["Date"])


print("###############################  remove extra spaceing from start and end (only valid for string datatype) ################")

# Remove leading and trailing spaces
df["Name"] = df["Name"].str.strip()



print("###############################  connver column into upper case  (only valid for string datatype )################")

# Convert product names to uppercase
df["Name"] = df["Name"].str.upper()













