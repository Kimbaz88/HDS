import streamlit as st
import numpy as np
import pickle

st.title('Preterm labor predictor')

st.write('Welcome to the preterm labor prediction app. Below you can check one or more boxes based on medical history and you will find the probability of a delivery before 37 weeks of gestation.')
#df = pd.read_csv('C://Users//kimam//Desktop//Insight//App//cleaned_df2')
#st.dataframe(df)
st.header('Pre-pregnancy')
st.write('Check if you have even had any of the following:')
#pre_p = ['Hypertension (high blood pressure)','Hypothyroidsm (under active thyroid)','Asthma']

ui_htn = st.checkbox('Hypertension (high blood pressure)')
#ui_th = st.checkbox('Hypothyroidsm (under active thyroid)')
ui_as = st.checkbox('Asthma')

st.header('During current pregnancy')
st.write('Check if you have experienced any of these during the current pregnancy:')
#p = ['Gestational hypertension (high blood pressure during pregnancy)','Gestational diabetes', 'Protein in urine','Anemia', 'Bladder or kidney infections','Vaginosis','Positive group B streptococcus swab','Hyperemesis (severe nausea)','RH disease','Currently smoking']
ui_ghtn = st.checkbox('Gestational hypertension (high blood pressure during pregnancy)')
ui_gd = st.checkbox('Gestational diabetes')
#ui_ur = st.checkbox('Protein in urine')
ui_kid = st.checkbox('Bladder or kidney infections')
ui_ane = st.checkbox('Anemia')
ui_vag = st.checkbox('Vaginosis')
#ui_grb = st.checkbox('Positive group B streptococcus swab')
#ui_nau = st.checkbox('Hyperemesis (severe nausea)')
#ui_rh = st.checkbox('RH disease')
ui_smok = st.checkbox('Currently smoking cigarettes')
ui_pre = st.checkbox('Preeclampsia')
ui_ins = st.checkbox('Are you covered by health insurance?')

#THYROID,HIGHBP_NOTPREG,ASTHMA,DIABETES,HIGHBP_PREG,PREECLAMPSIA,ANEMIA,KIDNEY,NAUSEA,RH_DISEASE,URINE,VAGINOSIS,GROUP_B,CIG_NOW
var_list = [ui_ins, ui_htn, ui_as, ui_gd, ui_ghtn,ui_kid, ui_ane, ui_vag, ui_smok, ui_pre]
#st.write(var_list)

var_list2 = np.asarray(var_list, dtype=np.float32)

var_list3 = var_list2.astype(int)
full_input = var_list3


pkl_filename = 'lgbm.pkl'
  
with open (pkl_filename,'rb') as file:
    pickle_model = pickle.load(file)


prediction = pickle_model.predict(full_input)
prediction_r = np.exp(prediction)
st.header('The preterm labor risk is')
st.write(prediction_r[0])




