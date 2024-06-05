import streamlit as st
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Установка черного фона
st.markdown(
    """
    <style>
    .reportview-container {
        background: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Загрузка данных из CSV-файла
mpg = pd.read_csv('C:\\MGTU\\6 semestr\\TMO\\auto-mpg.csv')

# Предварительная обработка данных
mpg = mpg[mpg['horsepower'] != '?']
mpg['horsepower'] = mpg['horsepower'].astype(float)
mpg = mpg.drop(columns=['car name'])
mpg = mpg.dropna()

# Выбор фичей и целевой переменной
X = mpg.drop(columns=['mpg'])
y = mpg['mpg']

# Разделение на обучающую и тестовую выборки
mpg_X_train, mpg_X_test, mpg_y_train, mpg_y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Функция для обучения BaggingRegressor
def train_bagging_regressor(n_estimators, max_features, max_samples, oob_score):
    estimator = DecisionTreeRegressor(random_state=1)
    br_mpg = BaggingRegressor(estimator=estimator, n_estimators=n_estimators, max_features=max_features,
                              max_samples=max_samples, oob_score=oob_score, random_state=10)
    br_mpg.fit(mpg_X_train, mpg_y_train)
    predictions = br_mpg.predict(mpg_X_test)
    mse = mean_squared_error(mpg_y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(mpg_y_test, predictions)
    return mse, rmse, mae


# Функция для обучения DecisionTreeRegressor
def train_decision_tree_regressor(max_depth, min_samples_split, min_samples_leaf, max_features):
    if max_depth == 'None':
        max_depth = None
    tree_regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=1)
    tree_regressor.fit(mpg_X_train, mpg_y_train)
    predictions = tree_regressor.predict(mpg_X_test)
    mse = mean_squared_error(mpg_y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(mpg_y_test, predictions)
    return mse, rmse, mae


st.title("Демонстрация моделей машинного обучения")

model_choice = st.selectbox("Выберите модель", ["Bagging Regressor", "Decision Tree Regressor"])

if model_choice == "Bagging Regressor":
    n_estimators = st.slider("n_estimators", 100, 2000, step=100)
    max_features = st.slider("max_features", 0.1, 1.0, step=0.1)
    max_samples = st.slider("max_samples", 0.1, 1.0, step=0.1)
    oob_score = st.checkbox("oob_score")

    mse, rmse, mae = train_bagging_regressor(n_estimators, max_features, max_samples, oob_score)

    st.write("Mean Squared Error: ", mse)
    st.write("Root Mean Squared Error: ", rmse)
    st.write("Mean Absolute Error: ", mae)

elif model_choice == "Decision Tree Regressor":
    max_depth = st.selectbox("max_depth", ['None', 3, 5, 7, 10, 12, 15])
    min_samples_split = st.slider("min_samples_split", 2, 20, step=1)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 10, step=1)
    max_features = st.selectbox("max_features", [None, 'sqrt', 'log2'])

    mse, rmse, mae = train_decision_tree_regressor(max_depth, min_samples_split, min_samples_leaf, max_features)

    st.write("Mean Squared Error: ", mse)
    st.write("Root Mean Squared Error: ", rmse)
    st.write("Mean Absolute Error: ", mae)

    # Отображение графика зависимости MSE от max_depth для DecisionTreeRegressor
    depths = [3, 5, 7, 10, 12, 15, None]
    mse_scores = []

    for depth in depths:
        if depth == 'None':
            depth = None
        mse, _, _ = train_decision_tree_regressor(depth, min_samples_split, min_samples_leaf, max_features)
        mse_scores.append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot([3, 5, 7, 10, 12, 15, None], mse_scores, marker='o', linestyle='--')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error')
    plt.title('Зависимость MSE от Max Depth')
    plt.grid(True)
    st.pyplot(plt)
