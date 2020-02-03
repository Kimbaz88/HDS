import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing

								
# Creating an empty Dataframe with column names only
#dfapp = pd.DataFrame(columns=['ANEMIA', 'ASTHMA', 'DIABETES', 'CIG_NOW', 'HIGHBP_NOTPREG','HIGHBP_PREECLAMPSIA','KIDNEY','MOM_INSURANCE','VAGINOSIS'])
#print("Empty Dataframe ", dfapp, sep='\n')

ANEMIA = []
ASTHMA = []
DIABETES = []
CIG_NOW = []
HIGHBP_NOTPREG = []
HIGHBP_PREECLAMPSIA = []
KIDNEY = []
MOM_INSURANCE = []
VAGINOSIS = []

st.title('Preterm labor predictor')

st.write('Welcome to the preterm labor prediction app. Below you can check one or more boxes based on medical history and find the risk class (low or high) of a delivery before 37 weeks of gestation.')
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

#var_list2 = np.asarray(var_list, dtype=np.float32)

#var_list3 = var_list2.astype(int)
#full_input = var_list3
#st.write(var_list2)
#st.write(var_list3)


ANEMIA.append(ui_ane)
ASTHMA.append(ui_as)
DIABETES.append(ui_gd)
CIG_NOW.append(ui_smok)
HIGHBP_NOTPREG.append(ui_htn)
HIGHBP_PREECLAMPSIA.append(ui_pre)
KIDNEY.append(ui_kid)
MOM_INSURANCE.append(ui_ins)
VAGINOSIS.append(ui_vag)

#df= pd.DataFrame({'ANEMIA': ANEMIA,'ASTHMA':ASTHMA, 'DIABETES':DIABETES,'CIG_NOW':CIG_NOW,'HIGHBP_NOTPREG':HIGHBP_NOTPREG,
                  #'HIGHBP_PREECLAMPSIA':HIGHBP_PREECLAMPSIA,'KIDNEY':KIDNEY,'MOM_INSURANCE':MOM_INSURANCE,'VAGINOSIS':VAGINOSIS})
results = pd.DataFrame([ANEMIA, ASTHMA, DIABETES, CIG_NOW, HIGHBP_NOTPREG, HIGHBP_PREECLAMPSIA, KIDNEY,MOM_INSURANCE,VAGINOSIS], index=["ANEMIA", "ASTHMA", "DIABETES", "CIG_NOW", "HIGHBP_NOTPREG", "HIGHBP_PREECLAMPSIA", "KIDNEY", "MOM_INSURANCE","VAGINOSIS"])
results = results.astype(int)
#results
# Create LabelEncoder Object and fit(),transform()
lbl = preprocessing.LabelEncoder()
lbl.fit(results)
sample = lbl.transform(results)
# Converted value by LabelEncoder
st.write(sample)
sample = np.reshape(sample,(1, sample.size))
sample


pkl_filename = 'lgbm.pkl'
pickle_in = open("lgbm.pkl","rb")
pickle_model = pickle.load(pickle_in)  
#with open (pkl_filename,'rb') as file:
    #pickle_model = pickle.load(file)
pred = pickle_model.predict(sample)

#convert into binary values
for i in pred:
    if (i >=0.5):
        predup=1 
    else:
        predup=0
   ##finda way to only ouput after clicking, and make it update when uncliking     
#prediction_r = np.exp(prediction)
st.header('The preterm labor risk is')
if predup == 0:
    st.write('Low')
else:
    st.write('High. Consider preventitive strategies and additional patient monitoring. The American College of Obstetricians and Gynecologists (ACOG) guidelines can be found here (link).')