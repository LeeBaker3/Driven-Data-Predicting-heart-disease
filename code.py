import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
y=pd.read_csv('train_labels.csv')

#joining features
allfeat = pd.concat([train, test],axis=0)
print(train.shape,test.shape,allfeat.shape)

#splitting features into ranges

allfeat['resting_blood_pressure'] = pd.qcut(allfeat['resting_blood_pressure'], 10)
allfeat['serum_cholesterol_mg_per_dl'] = pd.qcut(allfeat['serum_cholesterol_mg_per_dl'], 10)
allfeat['max_heart_rate_achieved'] = pd.qcut(allfeat['max_heart_rate_achieved'], 5)
allfeat['age'] = pd.qcut(allfeat['age'],5)

#one-hot encoding categorical variables
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['thal'], prefix='thal')],axis=1) 
allfeat=allfeat.drop(columns='thal') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['resting_ekg_results'],prefix='ekg')],axis=1) 
allfeat=allfeat.drop(columns='resting_ekg_results') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['chest_pain_type'],prefix='pain_type')],axis=1) 
allfeat=allfeat.drop(columns='chest_pain_type') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['max_heart_rate_achieved'],prefix='maxheartrate')],axis=1) 
allfeat=allfeat.drop(columns='max_heart_rate_achieved') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['age'],prefix='age')],axis=1) 
allfeat=allfeat.drop(columns='age') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['resting_blood_pressure'],prefix='restbp')],axis=1) 
allfeat=allfeat.drop(columns='resting_blood_pressure') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['serum_cholesterol_mg_per_dl'],prefix='serum_cholest')],axis=1) 
allfeat=allfeat.drop(columns='serum_cholesterol_mg_per_dl') 

allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['sex'],prefix='sex')],axis=1) 
allfeat=allfeat.drop(columns='sex') 


#renaming columns
allfeat.columns=['patient_id', 'slope_of_peak_exercise_st_segment', 'num_major_vessels', 'fasting_blood_sugar_gt_120_mg_per_dl', 'oldpeak_eq_st_depression', 'exercise_induced_angina', 'thal_fixed_defect', 'thal_normal', 'thal_reversible_defect', 'ekg_0', 'ekg_1', 'ekg_2','pain_type_1', 'pain_type_2', 'pain_type_3', 'pain_type_4', 'maxheartrate_70.999_128.8', 'maxheartrate_128.8_147.0', 'maxheartrate_147.0_159.0','maxheartrate_159.0_170.0', 'maxheartrate_170.0_202.0', 'age_28.999_45.0', 'age_45.0_52.0', 'age_52.0_58.0', 'age_58.0_62.2', 'age_62.2_77.0','restbp1','restbp2','restbp3','restbp4','restbp5','restbp6','restbp7','restbp8','restbp9','restbp10','serum_cholest_1','serum_cholest_2','serum_cholest_3','serum_cholest_4','serum_cholest_5','serum_cholest_6','serum_cholest_7','serum_cholest_8','serum_cholest_9','serum_cholest_10','fem','male']

train=allfeat[:][0:180]
test=allfeat[:][180:270]

#print(train.shape,test.shape,allfeat.info())
print('Training data...',train.shape)

X_train=train.drop(columns='patient_id')
X_test=test.drop(columns='patient_id')
y=y.drop(columns='patient_id')

param_grid = [{'C': np.arange(0.1, 10.1, 0.1)}] #set of trial values for min_child_weight
clf = GridSearchCV(SVC(probability=True), param_grid, cv=10, scoring= 'neg_log_loss',iid=True)
clf.fit(X_train,y)

probs=clf.predict_proba(X_test)
present_proba=np.delete(probs,0,axis=1)
present_proba=present_proba.flatten()
print(present_proba)

op=pd.DataFrame(data={'patient_id':test['patient_id'],'heart_disease_present':present_proba})
swaptitle=['patient_id','heart_disease_present']
op=op.reindex(columns=swaptitle)
op.to_csv('Gridsearch_SVC_submission.csv',index=False)




