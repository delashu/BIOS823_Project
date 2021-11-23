import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import joblib
import pickle
import xgboost

#set wide format
st.set_page_config(layout="wide")

#load in the training data
df = pd.read_pickle('scripts/dashboard/demo_training.pkl')

st.title("ICU Death Monitoring Dashboard")

# side bar text
st.sidebar.header("ICU Death Prediction Dashboard")
st.sidebar.text("This dashboard is created for \nhealthcare spcecialists in the \nIntensive Care Unit (ICU). \nThe dashboard allows the end user \nto explore previous ICU deaths. The \nuser can then input patient \ninformation typically collected \nduring an ICU stay. The dashboard \nwill output corresponding risk of \ndeath and useful algorithm metrics.")
st.sidebar.subheader("I. Exploratory Data Analysis:")
st.sidebar.text("The top of the dashboard allows \nthe user to perform exploratory \ndata analysis based on MIMIC demo \ndata. Select a continuous and a \ncategorical variable of interest to \nfirst visualize data by Death Status.")
st.sidebar.subheader("II. Predictive Modeling")
st.sidebar.text("Input patient information \ncollected during the ICU. The \ndashboard will then output \nthe patient's risk of death with \na corresponding low, medium, \nand high level. Prediction score \nreflects model trained on full MIMIC data.")
st.sidebar.text(" ")
st.sidebar.subheader("Authors:")
st.sidebar.text("Shusaku Asai, Yi Feng,\nSaahithi Rao, Michael Tang")

# rename for plotting purposes
df = df.rename(columns={'hospital_expire_flag': 'Death',
                       'los': 'Length of Stay'})

# rename outcome
df['Death'] = np.where(df['Death']==0, "Not Dead",
                                   np.where(df['Death'] == 1, "Dead", "Other"))

# output into column features in streamlit
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

#output px histogram into object
fig_cont = px.histogram(df, x=cont_select, color="Death", marginal="box",
                       title= str(cont_select) + str(': Histogram and Boxplot by Death Status'))
#create category counts with groupby statement for user input value
categ_counts = df.groupby(['Death',categ_select]).size().reset_index(name='counts')
#output go barplot with the grouped data above
barfig = go.Figure(data=[
    go.Bar(name='Not Dead',     
           x = categ_counts[categ_counts['Death'] == "Not Dead" ][categ_select],
           y = categ_counts[categ_counts['Death'] == "Not Dead" ].counts),
    go.Bar(name='Dead',     
           x = categ_counts[categ_counts['Death'] == "Dead" ][categ_select],
           y = categ_counts[categ_counts['Death'] == "Dead" ].counts)
])
# Change the bar mode and output into two columns
barfig.update_layout(barmode='group',title_text=str(categ_select) + str(': Barplot by Death'))
col3, col4 = st.columns((1, 1))
col3.plotly_chart(fig_cont)
col4.plotly_chart(barfig)

# user inputs
st.subheader(str("**II: Predictive Modeling**"))
st.markdown(str("Input patient information below to output risk of ICU death."))
#below lines give user input for the final model
col5, col6, col7, col8, col9= st.columns([1, 1, 1, 1,1])
los_select = col5.number_input('Length of Stay', value = 5)
ACE_select = col6.number_input('ACET325', value=0, step=1)
CAL_select = col7.number_input('CALG1I', value=0, step=1)
DW_select = col8.number_input('D5W1000', value=0, step=1)
DW2_select = col9.number_input('D5W250', value=0, step=1)
col10, col11, col12, col13, col14 = st.columns([1,1,1,1,1])
FURO_select = col10.number_input('FURO40I', value=0, step=1)
HEPA_select = col11.number_input('HEPA5I', value=0, step=1)
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
col34, col35, col36, col37, col38, col39 = st.columns([1,1,1,1,1,1])
diag_select =  col34.selectbox("Diagnosis", ['Other','Sepsis','Organ Failure','CV Failure','CNS Failure'])
fc_select =  col35.selectbox("First Care Unit", ['MICU','SICU','CCU','TSICU','CSRU'])
fw_select =  col36.selectbox("First Ward", ['52','23','Other'])
at_select =  col37.selectbox("Admit Type", ['EMERGENCY','ELECTIVE','URGENT'])
al_select =  col38.selectbox("Admit Location", ['EMERGENCY ROOM ADMIT','TRANSFER FROM HOSP/EXTRAM','CLINIC REFERRAL/PREMATURE','PHYS REFERRAL/NORMAL DELI','TRANSFER FROM SKILLED NUR'])
ins_select =  col39.selectbox("Insurance Status", ['Medicare','Private','Medicaid','Other'])
#make pandas dataframe from the user input with default values
predat = {'los': [los_select], 'ACET325': [ACE_select],'CALG1I': [CAL_select], 'D5W1000': [DW_select],
'D5W250': [DW2_select], 'FURO40I': [FURO_select],'HEPA5I': [HEPA_select], 'INSULIN': [INSU_select],
'KCL20P': [KCL_select], 'KCL20PM': [KCL2_select],'KCLBASE2': [KCLBASE_select], 'LR1000': [LR_select],
'MAG2PM': [MAG_select], 'METO25': [METO_select],'MORP2I': [MORP_select], 'NACLFLUSH': [NACL_select],
'NS1000': [NS_select], 'NS250': [NS2_select],'NS500': [NS5_select], 'VANC1F': [VANC_select],
'VANCOBASE': [VANCBASE_select], 'Dialysis': [DIAL_select],'Imaging': [IMAG_select], 'Intubation/Extubation': [INTUB_select],
'Invasive Lines': [INV_select], 'Peripheral Lines': [PERI_select],'Procedures': [PROC_select], 'Significant Events': [SIGNIF_select],
'Ventilation': [VENT_select], 'first_wardid_52': [0], 'first_wardid_Other': [0], 'first_careunit_CSRU': [0],
'first_careunit_MICU': [0], 'first_careunit_NICU': [0], 'first_careunit_SICU': [0], 'first_careunit_TSICU': [0],
'admission_type_EMERGENCY': [0], 'admission_type_NEWBORN': [0], 'admission_type_URGENT': [0], 
'admission_location_CLINIC REFERRAL/PREMATURE': [0],
'admission_location_EMERGENCY ROOM ADMIT': [0], 'admission_location_HMO REFERRAL/SICK': [0],
'admission_location_PHYS REFERRAL/NORMAL DELI': [0], 'admission_location_TRANSFER FROM HOSP/EXTRAM': [0], 
'admission_location_TRANSFER FROM OTHER HEALT': [0], 'admission_location_TRANSFER FROM SKILLED NUR': [0], 
'admission_location_TRSF WITHIN THIS FACILITY': [0], 'insurance_Medicare': [0],
'insurance_Other': [0], 'insurance_Private': [0], 'insurance_Self Pay': [0],
'diagnosis_CV Failure': [0], 'diagnosis_Organ Failure': [0], 
'diagnosis_Other': [0], 'diagnosis_Sepsis': [0]}

#make a pd dataframe
pred_df = pd.DataFrame(data=predat) 

# data frame variables need to match features of model
# set categorical features based on user input
if fw_select == "52":
    pred_df['first_wardid_52'] = 1
elif fw_select == "Other":
    pred_df['first_wardid_Other'] = 1
#first care unit
if fc_select == "CSRU":
    pred_df['first_careunit_CSRU'] = 1
elif fc_select == "MICU":
    pred_df['first_careunit_MICU'] = 1
elif fc_select == "SICU":
    pred_df['first_careunit_SICU'] = 1
elif fc_select == "TSICU":
    pred_df['first_careunit_TSICU'] = 1
#admission type
if at_select == "EMERGENCY":
    pred_df['admission_type_EMERGENCY'][0] = 1
elif at_select == "URGENT":
    pred_df['admission_type_URGENT'] = 1
#admission location
if al_select == "EMERGENCY ROOM ADMIT":
    pred_df['admission_location_EMERGENCY ROOM ADMIT'] = 1
elif al_select == "PHYS REFERRAL/NORMAL DELI":
    pred_df['admission_location_PHYS REFERRAL/NORMAL DELI'] = 1
elif al_select == "TRANSFER FROM HOSP/EXTRAM":
    pred_df['admission_location_TRANSFER FROM HOSP/EXTRAM'] = 1
elif al_select == "TRANSFER FROM SKILLED NUR":
    pred_df['admission_location_TRANSFER FROM SKILLED NUR'] = 1

#insurance
if ins_select == "Medicare":
    pred_df['insurance_Medicare'] = 1
elif ins_select == "Other":
    pred_df['insurance_Other'] = 1
elif ins_select == "Private":
    pred_df['insurance_Private'] = 1

#diagnosis
if diag_select == "CV Failure":
    pred_df['diagnosis_CV Failure'] = 1
elif diag_select == "Organ Failure":
    pred_df['diagnosis_Organ Failure'] = 1
elif diag_select == "Other":
    pred_df['diagnosis_Other'] = 1
elif diag_select == "Sepsis":
    pred_df['diagnosis_Sepsis'] = 1
    
#load best model after hypertuning using pickling
clf = joblib.load('scripts/dashboard/clf_best_full_data.pickle')

#make np dataframe into array
pred_array = pred_df.to_numpy()

#make prediction!
prediction = clf.predict_proba(pred_array[:1])[:,1]
# output death predictions as score from 0 to 100
model_score = round(100*prediction[0],2)

#output the score
st.subheader(str("**Patient Death Prediction Score**"))
st.metric(label="Model Risk Prediction Score", value=str(model_score) + str("%"))
#make the output pretty
st.markdown("**Risk Range:**  Low: <15,   Medium: >=15 & <20,    High: >=20")
if model_score>=20:
       st.error("Patient has high risk of death")
if model_score >=15 and model_score <20:
       st.warning("Patient has medium risk of death")
if model_score <15:
       st.success("Patient has low risk of death")
st.markdown("The prediction score is dervied from an XGBoost Random Forest Classifier (fit [here](https://github.com/delashu/BIOS823_Project/blob/main/scripts/modeling/full_data_modeling.ipynb))")
