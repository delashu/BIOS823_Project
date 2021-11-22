import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
#set wide format
st.set_page_config(layout="wide")
#load in the training data
df = pd.read_pickle('../modeling/demo_training.pkl')
st.sidebar.header("ICU Death Prediction Dashboard")
st.sidebar.text("This dashboard is created for \nhealthcare spcecialists in the \nIntensive Care Unit (ICU). \nThe dashboard allows the end user \nto explore previous ICU deaths. The \nuser can then input patient \ninformation typically collected \nduring an ICU stay. The dashboard \nwill output corresponding risk of \ndeath and useful algorithm metrics.")
st.sidebar.subheader("I. Exploratory Data Analysis:")
st.sidebar.text("The top of the dashboard allows \nthe user to perform exploratory \ndata analysis. Select a\ncontinuous and a categorical\nvariable of interest to first \nvisualize data by Death Status.")
st.sidebar.subheader("II. Predictive Modeling")
st.sidebar.text("Input patient information \ncollected during the ICU. The \ndashboard will then output \nthe patient's risk of death with \na corresponding low, medium, \nand high level. Model metrics \nare also provided.")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.subheader("Authors:")
st.sidebar.text("Shusaku Asai, Yi Feng,\nSaahithi Rao, Michael Tang")

#rename for plotting purposes:
df = df.rename(columns={'hospital_expire_flag': 'Death',
                       'los': 'Length of Stay'})


df['Death'] = np.where(df['Death']==0, "Not Dead",
                                   np.where(df['Death'] == 1, "Dead", "Other"))
#output into column features in streamlit
col1, col2 = st.columns((1, 1))
cont_select = col1.selectbox("Continuous", ['Length of Stay', 'ACET325',
       'CALG1I', 'D5W1000', 'D5W250', 'FURO40I', 'HEPA5I', 'INSULIN', 'KCL20P',
       'KCL20PM', 'KCLBASE2', 'LR1000', 'MAG2PM', 'METO25', 'MORP2I',
       'NACLFLUSH', 'NS1000', 'NS250', 'NS500', 'VANC1F', 'VANCOBASE',
       'Dialysis', 'Imaging', 'Intubation/Extubation', 'Invasive Lines',
       'Peripheral Lines', 'Procedures', 'Significant Events', 'Ventilation'])
#col2.subheader("Raw Data")
categ_select = col2.selectbox("Categorical", ['first_careunit', 'first_wardid',  'admission_type',
       'admission_location', 'insurance', 'diagnosis'])


fig_cont = px.histogram(df, x=cont_select, color="Death", marginal="box",
                       title= str(cont_select) + str(': Histogram and Boxplot by Death Status'))

categ_counts = df.groupby(['Death',categ_select]).size().reset_index(name='counts')
barfig = go.Figure(data=[
    go.Bar(name='Not Dead',     
           x = categ_counts[categ_counts['Death'] == "Not Dead" ][categ_select],
           y = categ_counts[categ_counts['Death'] == "Not Dead" ].counts),
    go.Bar(name='Dead',     
           x = categ_counts[categ_counts['Death'] == "Dead" ][categ_select],
           y = categ_counts[categ_counts['Death'] == "Dead" ].counts)
])

# Change the bar mode
barfig.update_layout(barmode='group',title_text=str(categ_select) + str(': Barplot by Death'))
col3, col4 = st.columns((1, 1))
col3.plotly_chart(fig_cont)
col4.plotly_chart(barfig)
