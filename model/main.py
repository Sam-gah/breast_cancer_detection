import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import sklearn as st
import numpy as np


def get_clean_data():
    df = pd.read_csv('D:\Minor_python\data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def plot_data(df):
    plot = df['diagnosis'].value_counts().plot(
        kind='bar', title="Class distributions \n(0: Benign | 1: Malignant)")
    plot.set_xlabel("Diagnosis")
    plot.set_ylabel("Frequency")
    plt.show()
def get_model():


    df = get_clean_data()

    # scale predictors and split data
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    # Train and evaluate each model using cross-validation
    best_model = None
    best_f1_score = 0
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        avg_f1_score = cv_scores.mean()
        print(f"{name} - Average F1-score: {avg_f1_score}")
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_model = model.fit(X, y)  # Fit the best model before returning

    print(f"Best Model: {best_model} with F1-score: {best_f1_score}")
    return best_model, scaler

def create_radar_chart(input_data):

    import plotly.graph_objects as go
    
    input_data = get_scaled_values_dict(input_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig
def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Cell Nuclei Details")
    slider_labels = [("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
                     ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
                     ("Smoothness (mean)",
                      "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
                     ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)",
                                                              "concave points_mean"),
                     ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)",
                                                            "fractal_dimension_mean"),
                     ("Radius (se)", "radius_se"), ("Texture (se)",
                                                    "texture_se"), ("Perimeter (se)", "perimeter_se"),
                     ("Area (se)", "area_se"), ("Smoothness (se)", "smoothness_se"),
                     ("Compactness (se)",
                      "compactness_se"), ("Concavity (se)", "concavity_se"),
                     ("Concave points (se)",
                      "concave points_se"), ("Symmetry (se)", "symmetry_se"),
                     ("Fractal dimension (se)",
                      "fractal_dimension_se"), ("Radius (worst)", "radius_worst"),
                     ("Texture (worst)", "texture_worst"), ("Perimeter (worst)",
                                                            "perimeter_worst"),
                     ("Area (worst)", "area_worst"), ("Smoothness (worst)",
                                                      "smoothness_worst"),
                     ("Compactness (worst)",
                      "compactness_worst"), ("Concavity (worst)", "concavity_worst"),
                     ("Concave points (worst)",
                      "concave points_worst"), ("Symmetry (worst)", "symmetry_worst"),
                     ("Fractal dimension (worst)", "fractal_dimension_worst")]

    input_data = {}

    for label, col in slider_labels:
        input_data[col] = st.sidebar.slider(
            label, float(data[col].min()), float(
                data[col].max()), float(data[col].mean())
        )

    return input_data
def get_scaled_values_dict(values_dict):
    # Define a Function to Scale the Values based on the Min and Max of the Predictor in the Training Data
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in values_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict
def display_predictions(input_data, model, scaler):
    import streamlit as st

    import numpy as np
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)

    st.subheader('Cell cluster prediction')
    st.write("The cell cluster is: ")

    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.write("<span class='diagnosis bright-green'>Benign</span>",
                 unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis bright-red'>Malignant</span>",
                 unsafe_allow_html=True)

    st.write("Probability of being benign: ",
             model.predict_proba(input_data_scaled)[0][0])
    st.write("Probability of being malignant: ",
             model.predict_proba(input_data_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
def create_app():
    import streamlit as st
    import numpy as np

    st.set_page_config(page_title="Breast Cancer Diagnosis",
                       page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")

    # load css
    with open("./assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)

    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    data = get_clean_data()
    input_data = create_input_form(data)

    best_model, scaler = get_model()
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = create_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        # Predict using the selected model
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        input_data_scaled = scaler.transform(input_array)
        prediction = best_model.predict(input_data_scaled)

        st.subheader('Cell cluster prediction')
        st.write("The cell cluster is: ")

        if prediction[0] == 0:
            st.write("<span class='diagnosis bright-green'>Benign</span>",
                     unsafe_allow_html=True)
        else:
            st.write("<span class='diagnosis bright-red'>Malignant</span>",
                     unsafe_allow_html=True)

        st.write("Probability of being benign: ",
                 best_model.predict_proba(input_data_scaled)[0][0])
        st.write("Probability of being malignant: ",
                 best_model.predict_proba(input_data_scaled)[0][1])

        st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
def main():
    create_app()

if __name__ == '__main__':
    main()
