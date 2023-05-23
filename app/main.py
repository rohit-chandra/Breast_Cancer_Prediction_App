import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys

def get_clean_data():
    data = pd.read_csv("../data/data.csv")
    
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    # print(data.head())
    # malignant = 1; benign = 0
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = get_clean_data()
    
    # Create a list of tuples containing the labels and keys
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_dict = {}

    for label, key in slider_labels:
        # value is the default value on the slider
        # st.sidebar.slider returns the value of the slider. We store it in input_dict as value for each column as key
        input_dict[key] = st.sidebar.slider(label, min_value = float(0), max_value = float(data[key].max()), value = float(data[key].mean()))
        
    return input_dict


def get_scaled_values(input_dict):
    """scale input values between 0 and 1
    
    Args:
        input_dict (_type_): _description_
    
    Returns:
        dict: scaled values
    """
    data = get_clean_data()
    
    X = data.drop(["diagnosis"], axis=1)
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict



def get_radar_char(input_data):
    # scale the input data
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                    'Smoothness', 'Compactness', 
                    'Concavity', 'Concave Points',
                    'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()
    
    # r = radial values
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))
    
    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig



def add_predictions(input_data):
    reconstructed_model = pickle.load(open("../model/model.pkl", "rb"))
    reconstructed_scaler = pickle.load(open("../model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = reconstructed_scaler.transform(input_array)
    
    prediction = reconstructed_model.predict(input_array_scaled)
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)
        
    # probability of the prediction
    st.write("Probability of being benign:", reconstructed_model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malignant:", reconstructed_model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist medical professionals in making a diagonsis, but should not be used as a sibstitute for professional diagnosis. ")

def main():
    # set page config
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")
    
    temp_file = "../assets/style2.css"
    
    # import CSS as markdown file
    with open("../assets/style2.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    # add the sidebar
    input_data = add_sidebar()
    
    # st.write(input_data)
    # Create a container and add elements inside it
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app leverages machine learning to predict the **Breast Cancer** based on the input measurements. You can update the paramters using the sliders in the sidebar")
    # Create columns with size ratio 4:1
    # 1st col = 4 times as big as 2nd col
    # 2nd col = 1
    col1, col2 = st.columns([4, 1])
    
    # to add contents inside col1, use "with col1:"
    with col1:
        # add radar chart in the 1st column
        radar_chart = get_radar_char(input_data)
        # plot the chart
        st.plotly_chart(radar_chart)
    with col2:
        # display the prediction in the 2nd column
        add_predictions(input_data)
        
        

if __name__ == '__main__':
    main()