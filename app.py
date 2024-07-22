import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    dataset_path = r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\SALES PRIDECTION\advertising (1).csv"
    return pd.read_csv(dataset_path)

df = load_data()

# Prepare the data
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Sales Prediction App')

st.sidebar.header('User Input')
tv = st.sidebar.slider('TV Advertising Budget', float(X['TV'].min()), float(X['TV'].max()), float(X['TV'].mean()))
radio = st.sidebar.slider('Radio Advertising Budget', float(X['Radio'].min()), float(X['Radio'].max()), float(X['Radio'].mean()))
newspaper = st.sidebar.slider('Newspaper Advertising Budget', float(X['Newspaper'].min()), float(X['Newspaper'].max()), float(X['Newspaper'].mean()))

user_input = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})

st.subheader('User Input:')
st.write(user_input)

# Make prediction
prediction = model.predict(user_input)

st.subheader('Sales Prediction:')
st.write(f"${prediction[0]:.2f}")

# Model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader('Model Performance:')
st.write(f"Root Mean Squared Error: {rmse:.2f}")
st.write(f"R-squared Score: {r2:.2f}")

# Visualizations
st.subheader('Visualizations')

# Actual vs Predicted Sales
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales")
st.pyplot(fig)

# Feature Importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='feature', y='importance', data=feature_importance, ax=ax)
ax.set_xlabel("Features")
ax.set_ylabel("Absolute Coefficient Value")
ax.set_title("Feature Importance")
plt.xticks(rotation=45)
st.pyplot(fig)

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# Pairplot
st.write("Pairplot of Features and Sales")
pairplot = sns.pairplot(df, vars=['TV', 'Radio', 'Newspaper', 'Sales'], height=2)
st.pyplot(pairplot.fig)