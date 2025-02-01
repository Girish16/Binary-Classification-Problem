# Remove leading/trailing spaces in string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# For simplicity, drop rows with missing values (in production, consider imputation)
df.dropna(inplace=True)

print("Dataset shape after dropping missing values:", df.shape)
