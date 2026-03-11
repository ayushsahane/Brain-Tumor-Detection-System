import streamlit as st
from .components import title
from .utils import set_css
import streamlit.components.v1 as html_components

def main():
    # Display the title using an HTML component
    html_components.html(title())
    
    # Apply custom CSS from an external file
    set_css("pages/css/streamlit.css")
    
    # Display the "About" section with centered heading
    st.markdown("<h4 style='text-align: center;'>About</h4>", unsafe_allow_html=True)
    about_text = """
    This app is written in Python 3 using the TensorFlow library 
    and Keras to build a neural network for image classification.
    Keras is a high-level API for TensorFlow. In this case,
    a Convolutional Neural Network (CNN) was created using multiple
    layers of convolutional and pooling layers. The network was trained
    using datasets found on Kaggle. Importantly, the training data was
    cleaned using the [mask.py](https://github.com/ayushsahane/Brain-Tumor-Detection-System/blob/main/mask.py)
    script and cropped to remove unnecessary backgrounds. After training,
    the model and its weights were saved to a file. The model can be
    loaded using the [get_model(num)](https://github.com/ayushsahane/Brain-Tumor-Detection-System/blob/main/predictor.py)
    function, where 'num' specifies the model to be loaded. This interface is
    built using Streamlit and allows for fast and intuitive analysis of the given data.
    """
    st.write(about_text)
    
    # Display the "Abstract" section with centered heading
    st.markdown("<h4 style='text-align: center;'>Abstract</h4>", unsafe_allow_html=True)
    abstract_text = """
    Brain cancers (e.g., glioblastomas) are some of the deadliest types of 
    cancers and are difficult to treat due to their anatomical 
    location. Early tumor detection is crucial for a good prognosis, but often 
    the diagnosis is difficult because the tumor is too small and not easily 
    detectable. Traditional diagnostic techniques include brain scans (i.e., MRI), 
    which can be time-consuming for doctors to analyze. An alternative, efficient 
    technique for diagnosis (analysis of the MRI) is the use of machine learning, 
    which can provide fast and accurate detection with an image classifier. 
    Here, we used open-source MRI datasets that were trimmed and resized for accurate
    results. We implemented TensorFlow's Convolutional Neural Networks (CNN) as the 
    architecture for our model. The images used in our algorithm consisted of 46% with tumors 
    and 54% that were non-cancerous. The program takes about 3.47 seconds to load the model and 
    produce predictions. Our model has a validation loss of 0.122 and a 99.50% maximum validation accuracy. 
    Although our model focuses on brain tumors, it can be extended to other types of 
    cancers diagnosed using similar methods (e.g., MRI). 
    The source code is available at [GitHub](https://github.com/ayushsahane/Brain-Tumor-Detection-System).
    """
    st.write(abstract_text)
