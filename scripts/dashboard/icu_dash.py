import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
#set wide format
st.set_page_config(layout="wide")
#read in training demo dat 
#load in the training data
df = pd.read_pickle('../modeling/demo_training.pkl')