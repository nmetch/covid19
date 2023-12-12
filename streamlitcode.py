import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC


def file_upload():
    st.header("Step 1: Upload your CSV file")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            if st.checkbox("Preview DataFrame"):
                st.dataframe(df.head())
            return df
        except Exception as e:
            st.warning("Unable to read file, please upload a valid CSV file.")
            return None
    else:
        return None


def preprocessing(df):
    st.header("Step 2: Preprocessing")

    # Suggest best preprocessing steps based on data
    st.write("Suggested preprocessing steps based on data:")
    suggested_steps = []
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        suggested_steps.append("Standard Scaling")
    if df.isnull().sum().sum() > 0:
        suggested_steps.append("Imputation")

    # Allow user to select preprocessing steps
    selected_steps = st.multiselect(
        "Select preprocessing steps", suggested_steps)

    # Perform selected preprocessing steps
    if "Standard Scaling" in selected_steps:
        scaler = StandardScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
            df[df.select_dtypes(include=[np.number]).columns])
    if "Imputation" in selected_steps:
        df = df.fillna(method='ffill')

    # Return preprocessed dataframe and selected preprocessing steps
    return df, selected_steps


def feature_selection(df):
    st.header("Step 3: Feature Selection")

    # Show available features
    st.write("Available features:")
    st.write(df.columns)

    # Suggest best features based on correlation with target variable
    st.write("Suggested features based on correlation with target variable:")
    corr = df.corr()
    corr_target = abs(corr["target"])
    best_features = corr_target[corr_target >= 0.5].index.tolist()
    st.write(best_features)

    # Allow user to select features
    selected_features = st.multiselect(
        "Select features", df.columns, default=best_features)

    # Return selected features and correlation matrix
    return df[selected_features, corr]


def model_selection(X, y):
    st.header("Step 3: Model Selection")

    # Create a table to display the accuracy of different models
    st.subheader("Model Comparison")
    models = {"Random Forest": RandomForestClassifier(),
              "Logistic Regression": LogisticRegression(),
              "Support Vector Machine": SVC()}
    scores = {}
    pipelines = {}
    for name, model in models.items():
        pipeline = make_pipeline(
            StandardScaler(),
            FunctionTransformer(preprocessing, validate=False),
            FunctionTransformer(feature_selection, validate=False),
            model
        )

        param_grid = {}
        if name == "Random Forest":
            param_grid = {'randomforestclassifier__n_estimators': [10, 50, 100, 500],
                          'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2']}
        elif name == "Logistic Regression":
            param_grid = {'logisticregression__C': [0.1, 1, 10, 100]}
        elif name == "Support Vector Machine":
            param_grid = {'svc__C': [0.1, 1, 10, 100],
                          'svc__kernel': ['linear', 'rbf']}
        grid = GridSearchCV(pipeline, param_grid, cv=5)
        grid.fit(X, y)
        score = grid.best_score_
        scores[name] = score
        pipelines[name] = grid.best_estimator_

    df_scores = pd.DataFrame(scores.items(), columns=["Model", "Accuracy"])
    st.table(df_scores)

    # Suggest the best model based on the data
    best_model = max(scores, key=scores.get)
    st.subheader("Suggested Model")
    st.write("Based on the data, we suggest using", best_model)

    # Allow the user to select a model
    model_choice = st.selectbox("Select a model", list(models.keys()))

    # Train and predict using the selected model
    pipeline = pipelines[model_choice]
    pipeline.fit(X, y)

    # Display the accuracy and predicted results
    st.subheader("Accuracy")
    st.write("Accuracy:", pipeline.score(X, y))
    y_pred = pipeline.predict(X)
    st.subheader("Predicted Results")
    st.write(y_pred)

    # Display any relevant graphs or plots
    st.subheader("Graphs/Plots")
    if model_choice == "Random Forest":
        feature_importance = pipeline.steps[-1][1].feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        features = X.columns
        plt.figure()
        plt.title("Feature Importance")
        plt.bar(range(X.shape[1]), feature_importance[indices])
        plt.xticks(range(X.shape[1]), features[indices], rotation=90)
        plt.show()
        st.pyplot()

    if model_choice == "Logistic Regression":
        coef = pipeline.steps[-1][1].coef_
        features = X.columns
        plt.figure()
        plt.title("Feature Coefficients")
        plt.bar(range(X.shape[1]), coef[0])
        plt.xticks(range(X.shape[1]), features, rotation=90)
        plt.show()
        st.pyplot()

    plt.figure()
    plt.title("Confusion Matrix")
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt="g")
    plt.show()
    st.pyplot()

    return model_choice, pipeline


def predict(data, pipeline, dataset):
    """
    Make predictions using the trained pipeline on new data.

    Parameters:
    -----------
    data : pandas DataFrame
        Input data to make predictions on.
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline object to use for making predictions.
    dataset : str
        Name of the dataset used for training the model.

    Returns:
    --------
    str
        Formatted output with predicted values.
    """

    # Apply preprocessing steps to the input data
    preprocessed_data = pipeline[:-1].transform(data)

    # Make predictions using the trained model
    predictions = pipeline[-1].predict(preprocessed_data)

    # Convert units and format the output based on the dataset
    if dataset == "iris":
        output = f"The predicted species is {predictions[0]}."
    elif dataset == "diabetes":
        output = f"The predicted diabetes outcome is {'positive' if predictions[0] == 1 else 'negative'}."
    elif dataset == "wine":
        output = f"The predicted wine quality is {predictions[0]:.2f}."
    else:
        output = f"The predicted value is {predictions[0]:.2f} {data.unit}."

    return output


def main():
    st.set_page_config(page_title="Machine Learning App")

    st.title("Machine Learning App")

    # Ask user to upload a file
    uploaded_file = file_upload()

    # Preprocess the data
    df, target_col = preprocessing(uploaded_file)

    # Perform feature selection and model selection
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model_choice, pipeline = model_selection(X, y)

    # Display the suggested model and accuracy
    st.subheader("Model")
    st.write(f"The suggested model is {model_choice}.")
    st.write(f"The accuracy of the model is {pipeline.score(X, y):.2f}.")

    # Ask user to enter new data to make a prediction
    st.subheader("Make a Prediction")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(
            f"Enter the value for {col}", format="%f")

    # Convert the input data to a DataFrame and make a prediction
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = predict(input_df, pipeline)

    # Display the prediction
    st.subheader("Prediction")
    st.write(prediction)
