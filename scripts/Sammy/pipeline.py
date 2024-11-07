#libraries needed 
import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from libs.data_lib.eda import create_symbol_dataframes
from libs.io_lib.paths import DATA_DIR
from libs.plot_lib.data import plot_feature_evolution
from libs.data_lib.display import display_symbol_timeline, get_symbol_statistics



file_path = DATA_DIR / 'train.parquet' / 'partition_id=0' / 'part-0.parquet'
# Read the data
data = pd.read_parquet(file_path, engine='pyarrow')

# Separate target from predictors
y = data.responder_6
X = data.drop(['responder_6'], axis=1)




# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


print(X_train.head())

# Step 1: Define preprocessing step

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Step 2: Define a model (here this the model of the template I took but obviously for our case the baseline will be different)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Step 3: Create and evaluate the pipeline 

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)
                           ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model using R²
score = r2_score(y_valid, preds)
print('R²:', score)