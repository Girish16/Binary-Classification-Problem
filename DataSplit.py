from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
