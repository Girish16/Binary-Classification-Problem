from sklearn.preprocessing import LabelEncoder

# Convert target to binary: '>50K' -> 1 and '<=50K' -> 0
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Identify categorical and numerical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('income')  # Exclude the target

# Use one-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Dataset shape after encoding:", df_encoded.shape)
