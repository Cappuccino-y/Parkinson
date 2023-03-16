
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))

    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)

    pos_ind = dem != 0
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]

    return 100 * np.mean(smap)

cur='/kaggle/input/amp-parkinsons-disease-progression-prediction/'
proteins = pd.read_csv(cur+'train_proteins.csv')
peptides = pd.read_csv(cur+'train_peptides.csv')
peptides_test = pd.read_csv(cur+'example_test_files/test_peptides.csv')
proteins_test = pd.read_csv(cur+'example_test_files/test_proteins.csv')
sample_test=pd.read_csv(cur+'example_test_files/sample_submission.csv')

clinical = pd.read_csv(cur+'train_clinical_data.csv')
test = pd.read_csv(cur+'example_test_files/test.csv')

clinical_data=clinical.drop(['upd23b_clinical_state_on_medication', 'patient_id'], axis=1)

pep_prop_test = pd.merge(proteins_test,peptides_test,on=['visit_id', 'visit_month', 'patient_id', 'UniProt'],how='left')
pep_prop_test['pep_per_pro']=pep_prop_test['PeptideAbundance'] / pep_prop_test['NPX']
pep_prop_test=pep_prop_test.drop(['patient_id', 'visit_month'], axis=1).pivot(index=['visit_id'],
                                                                    columns=['Peptide'],
                                                                    values=['pep_per_pro'])
pep_prop_test.columns = pep_prop_test.columns.droplevel()
pep_prop_test=pep_prop_test.reset_index()

df = pep_prop_test.set_index('visit_id')
df.iloc[:,4]=df.iloc[:,4].fillna(0)
df_nan_index=df.isna().sum(axis=1)
df=df.drop(df[ df_nan_index>700 ].index,axis=0)
df=df.drop(df.columns[df.mean()==1.0],axis=1)
df=df.drop(df.columns[(df.isna().sum()>100)],axis=1)
test=df


pep_prop_train = pd.merge(proteins,peptides,on=['visit_id', 'visit_month', 'patient_id', 'UniProt'],how='left')
pep_prop_train['pep_per_pro']=pep_prop_train['PeptideAbundance'] / pep_prop_train['NPX']
pep_prop_train=pep_prop_train.drop(['patient_id', 'visit_month'], axis=1).pivot(index=['visit_id'],
                                                                    columns=['Peptide'],

                                                                  values=['pep_per_pro'])

pep_prop_train.columns = pep_prop_train.columns.droplevel()
pep_prop_train=pep_prop_train.reset_index()

df = pd.merge(clinical_data, pep_prop_train, on="visit_id", how="left")
df = df.set_index('visit_id')
df.iloc[:,4]=df.iloc[:,4].fillna(0)
df_nan_index=df.isna().sum(axis=1)
df=df.drop(df[ df_nan_index>700 ].index,axis=0)
df=df.drop(df.columns[df.mean()==1.0],axis=1)
df=df.drop(df.columns[(df.isna().sum()>100)],axis=1)

df=pd.concat([df.iloc[:,:5],df[df.columns.intersection(test.columns)]],axis=1)
test=test[df.columns.intersection(test.columns)].fillna(df.mean())
df=df.fillna(df.mean())

data=df.iloc[:,5:]
pca=PCA(n_components=256)
new_data=pca.fit_transform(data)
test_data=pca.transform(test)
print(pca.explained_variance_ratio_.sum())

ss=StandardScaler()
new_data=ss.fit_transform(new_data)
test_data=ss.transform(test_data)
test_data=pd.DataFrame(test_data,columns=['pca_'+ str(i) for i in range(test_data.shape[1])],index=test.index)
test_data['visit_month']=test_data.index.map(lambda x: x.split('_')[1])
new_data=pd.DataFrame(new_data,columns=['pca_'+ str(i) for i in range(new_data.shape[1])],index=df.index)
test_data['visit_month']=test_data.index.map(lambda x: int(x.split('_')[1]))
test_data=test_data[test_data.columns[:-1].insert(0,test_data.columns[-1])]
test_data=test_data.set_index(test_data.index.map(lambda x:x.split('_')[0]))
train_data=pd.concat([df.iloc[:,:5],new_data],axis=1)


xg_reg = MultiOutputRegressor( XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10))
xg_reg.fit(np.concatenate([train_data.iloc[:,0:1],new_data],axis=1),train_data.iloc[:,1:5])

def estimates(targets):
    input=test_data.loc[[targets[2]]]
    input.iloc[:,0]+=targets[0]
    return xg_reg.predict(input)[0][targets[1]-1]

import amp_pd_peptide
env = amp_pd_peptide.make_env()   # initialize the environment
iter_test = env.iter_test() 

# The API will deliver four dataframes in this specific order:
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
    # This maps the correct value estimate to each line in sample_submission
    targets = sample_submission.prediction_id.str.split('_').apply(lambda x: (int(x[1]) + int(x[5]), int(x[3]),x[0]))
    sample_submission['rating'] =targets.map(estimates) 
    env.predict(sample_submission)

