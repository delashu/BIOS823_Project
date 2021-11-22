import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
#set wide format
st.set_page_config(layout="wide")
#load in the training data
df = pd.read_pickle('demo_training.pkl')
st.sidebar.header("ICU Death Prediction Dashboard")
st.sidebar.text("This dashboard is created for \nhealthcare spcecialists in the \nIntensive Care Unit (ICU). \nThe dashboard allows the end user \nto explore previous ICU deaths. The \nuser can then input patient \ninformation typically collected \nduring an ICU stay. The dashboard \nwill output corresponding risk of \ndeath and useful algorithm metrics.")
st.sidebar.subheader("I. Exploratory Data Analysis:")
st.sidebar.text("The top of the dashboard allows \nthe user to perform exploratory \ndata analysis. Select a\ncontinuous and a categorical\nvariable of interest to first \nvisualize data by Death Status.")
st.sidebar.subheader("II. Predictive Modeling")
st.sidebar.text("Input patient information \ncollected during the ICU. The \ndashboard will then output \nthe patient's risk of death with \na corresponding low, medium, \nand high level. Model metrics \nare also provided.")
st.sidebar.text(" ")
st.sidebar.subheader("Authors:")
st.sidebar.text("Shusaku Asai, Yi Feng,\nSaahithi Rao, Michael Tang")

#rename for plotting purposes:
df = df.rename(columns={'hospital_expire_flag': 'Death',
                       'los': 'Length of Stay'})


df['Death'] = np.where(df['Death']==0, "Not Dead",
                                   np.where(df['Death'] == 1, "Dead", "Other"))
#output into column features in streamlit
st.subheader(str("**I: Exploratory Data Analysis**"))
st.markdown(str("Pick a continuous and categorical variable to explore from the dropdown menu based on historic ICU data."))
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

st.subheader(str("**II: Predictive Modeling**"))
st.markdown(str("Input patient information below."))
#empty_left, contents_left, contents_right, empty_right = st.beta_columns([1, 1.5, 1.5, 1])
col5, col6, col7, col8, col9= st.columns([1, 1, 1, 1,1])
los_select = col5.number_input('Length of Stay', value = 5)
ACE_select = col6.number_input('ACET325', value=0, step=1)
CAL_select = col7.number_input('CALG1I', value=0, step=1)
DW_select = col8.number_input('D5W1000', value=0, step=1)
DW2_select = col9.number_input('D5W250', value=0, step=1)

col10, col11, col12, col13, col14 = st.columns([1,1,1,1,1])
FURO_select = col10.number_input('FURO40I', value=0, step=1)
FURO_select = col11.number_input('HEPA5I', value=0, step=1)
INSU_select = col12.number_input('INSULIN', value=0, step=1)
KCL_select = col13.number_input('KCL20P', value=0, step=1)
KCL2_select = col14.number_input('KCL20PM', value=0, step=1)


col15, col16, col17, col18, col19 = st.columns([1,1,1,1,1])
KCLBASE_select = col15.number_input('KCLBASE2', value=0, step=1)
LR_select = col16.number_input('LR1000', value=0, step=1)
MAG_select = col17.number_input('MAG2PM', value=0, step=1)
METO_select = col18.number_input('METO25', value=0, step=1)
MORP_select = col19.number_input('MORP2I', value=0, step=1)

col20, col21, col22, col23, col24, col25 = st.columns([1,1,1,1,1,1])
NACL_select = col20.number_input('NACLFLUSH', value=0, step=1)
NS_select = col21.number_input('NS1000', value=0, step=1)
NS2_select = col22.number_input('NS250', value=0, step=1)
NS5_select = col23.number_input('NS500', value=0, step=1)
VANC_select = col24.number_input('VANC1F', value=0, step=1)
VANCBASE_select =  col25.number_input('VANCOBASE', value=0, step=1)


col26, col27, col28, col29, col30, col31, col32, col33 = st.columns([1,1,1,1,1,1,1,1])
DIAL_select = col26.number_input('Dialysis', value=0, step=1)
IMAG_select = col27.number_input('Imaging', value=0, step=1)
INTUB_select = col28.number_input('Intubation/Extubation', value=0, step=1)
INV_select = col29.number_input('Invasive Lines', value=0, step=1)
PERI_select = col30.number_input('Peripheral Lines', value=0, step=1)
PROC_select = col31.number_input('Procedures', value=0, step=1)
SIGNIF_select = col32.number_input('Significant Events', value=0, step=1)
VENT_select = col33.number_input('Ventilation', value=0, step=1)