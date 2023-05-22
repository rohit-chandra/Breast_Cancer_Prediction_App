import streamlit as st
import pickle
import pandas as pd


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
        
        
def main():
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")
    
    
    add_sidebar()
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
        st.write("## Input Parameters")
    
    with col2:
        st.write("")
        
        
        
        
if __name__ == '__main__':
    main()