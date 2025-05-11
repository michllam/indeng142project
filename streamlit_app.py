import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title='Student Performance EDA',
    page_icon='📊'
)

st.title("📊 Exploratory Data Analysis: Student Performance")

# Load dataset
data_path = 'data/student-por.csv'
df_raw = pd.read_csv(data_path)

# Drop G1 and G2
eda_df = df_raw.drop(['G1', 'G2'], axis=1).copy()

# Encode categorical variables for EDA
for col in eda_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    eda_df[col] = le.fit_transform(eda_df[col])

# Correlation Matrix
st.header("📈 Correlation Matrix")
fig1, ax1 = plt.subplots(figsize=(14, 10))
corr = eda_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax1)
plt.title('Correlation Matrix')
st.pyplot(fig1)

# Interactive Box Plot Section
st.header("📦 Compare Any Feature vs Final Grade")

# Create engineered features
if 'alcohol_avg' not in df_raw.columns:
    df_raw['alcohol_avg'] = (df_raw['Dalc'] + df_raw['Walc']) / 2
if 'ParentEdu' not in df_raw.columns:
    df_raw['ParentEdu'] = (df_raw['Medu'] + df_raw['Fedu']) / 2

# Select features for comparison
box_features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'alcohol_avg', 'ParentEdu']
selected_feature = st.selectbox("Choose a feature to compare with final grades (G3):", box_features)

# Draw interactive box plot
fig, ax = plt.subplots()
sns.boxplot(x=selected_feature, y='G3', data=df_raw, ax=ax)
plt.title(f'G3 vs {selected_feature}')
st.pyplot(fig)


# Pair Plot Section
st.header("🔗 Pair Plot of Key Features")
selected_pair_features = st.multiselect("Select features to include in the pair plot:", eda_df.columns.tolist(), default=['G3', 'studytime', 'failures'])
if len(selected_pair_features) >= 2:
    pair_fig = sns.pairplot(eda_df[selected_pair_features])
    st.pyplot(pair_fig)
else:
    st.info("Please select at least two features to generate a pair plot.")

# Classifier Section
st.header("🤖 Predict Academic Risk")
df = df_raw.copy()
df['study_efficiency'] = df['studytime'] / (df['failures'] + 1)
if 'G3' in df.columns:
    df['G3_binary'] = (df['G3'] < 10).astype(int)
else:
    st.error("'G3' column not found in the dataset. Please check the data loading and preprocessing steps.")
df = df.drop(columns=['G1', 'G2', 'G3'])
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['G3_binary'], errors='ignore')
y = df['G3_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train classifiers
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_model.fit(X_resampled, y_resampled)

ada_model = AdaBoostClassifier(learning_rate=1, random_state=42)
ada_model.fit(X_resampled, y_resampled)

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gb_model.fit(X_resampled, y_resampled)

# Sidebar model selector
model_choice = st.sidebar.selectbox("Choose a model for prediction:", ["Random Forest", "AdaBoost", "Gradient Boosting"])
model = rf_model if model_choice == "Random Forest" else ada_model if model_choice == "AdaBoost" else gb_model

# Input section
st.subheader("Input Features")

# Create dynamic feature inputs for top features
input_data = {}

# Manually add sliders or selectors for high-importance features
input_data['studytime'] = st.slider("Study Time", 1, 4, 2)
input_data['failures'] = st.slider("Number of Failures", 0, 3, 1)
input_data['alcohol_avg'] = st.slider("Average Alcohol (1-5)", 1.0, 5.0, 2.5)
input_data['ParentEdu'] = st.slider("Parent Education (0-4)", 0.0, 4.0, 2.0)
input_data['absences'] = st.slider("Absences", 0, 50, 4)
input_data['goout'] = st.slider("Going Out (1-5)", 1, 5, 3)
input_data['freetime'] = st.slider("Free Time (1-5)", 1, 5, 3)
input_data['health'] = st.slider("Health (1-5)", 1, 5, 3)
input_data['internet_yes'] = 1 if st.checkbox("Has Internet Access", True) else 0
input_data['higher_yes'] = 1 if st.checkbox("Plans Higher Education", True) else 0

input_data = pd.DataFrame([input_data])

# Ensure input columns match model
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[X.columns]

prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0][1]
label = '🚨 At Risk' if prediction == 1 else '✅ Not At Risk'
st.subheader(f"Prediction: {label}")
st.caption(f"Probability of being At Risk: {proba:.2f}")
