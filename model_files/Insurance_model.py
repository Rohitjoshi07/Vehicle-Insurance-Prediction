import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

def load():
    train= pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train,test

#train_map
def mapping(train):
    # mapping
    gender = {'Male': 1, 'Female': 0}
    vehical_age = {'1-2 Year': 2, "< 1 Year": 1, '> 2 Years': 0}
    vehical_damage = {'Yes': 1, 'No': 0}
    train['Gender'] =train['Gender'].map(gender)
    train['Vehicle_Damage'] = train['Vehicle_Damage'].map(vehical_damage)
    train['Vehicle_Age'] = train['Vehicle_Age'].map(vehical_age)

    #feature engineering
    train['log_premium'] = train['Annual_Premium'].apply(lambda x: np.log(x))
    train['Gender_premium'] = train.groupby(['Gender'])['log_premium'].transform('mean')
    train['Vehicle_Damage_premium'] = train.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    train['Gender_premium'] = train.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    train['Vehicle_Age_premium'] = train.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    train['Previously_Insured_premium'] = train.groupby(['Previously_Insured'])['log_premium'].transform('mean')
    return train


#using SMOTE(oversampling)
def smote(train):
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    oversampled_trainX, oversampled_trainY = sm.fit_sample(train.drop('Response', axis=1), train['Response'])
    oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
    bins_over = pd.qcut(oversampled_train['Age'],5,[0,1,2,3,4])
    oversampled_train['Age_bin'] = bins_over
    params = [i for i in oversampled_train.columns if i not in ['id','Region_Code','Response','Annual_Premium','Age']]
    trainx_over,testx_over,trainy_over,testy_over = train_test_split(oversampled_train[params],oversampled_train['Response'],test_size=0.10,random_state=42,shuffle=True)
    return trainx_over,testx_over,trainy_over,testy_over


#model development
def model(trainx_over,trainy_over,testx_over,testy_over):
    cat_feature =['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Age_bin']

    model3 = CatBoostClassifier(learning_rate=0.11,max_depth=7,n_estimators=1000,verbose=False,cat_features=cat_feature)
    model3.fit(trainx_over,trainy_over)
    predict_over_cat = model3.predict(testx_over)
    print(model3.score(testx_over,testy_over))
    print(classification_report(testy_over,predict_over_cat))

    #roc-auc score

    yhat1 = model3.predict_proba(testx_over)
    # retrieve just the probabilities for the positive class
    pos_probs1 = yhat1[:,1]
    roc_cat = roc_auc_score(testy_over, pos_probs1)
    print(f"Catboost roc_auc score is : {roc_cat}")
    return model3

#dumping model
def save_model(model3):
    pickle.dump(model3,open("cat_model.pkl",'wb'))




def preprocessing2(data):
    data['log_premium'] = data['Annual_Premium'].apply(lambda x: np.log(x))
    data['Gender_premium'] = data.groupby(['Gender'])['log_premium'].transform('mean')
    data['Vehicle_Damage_premium'] = data.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    data['Gender_premium'] = data.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    data['Vehicle_Age_premium'] = data.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    data['Previously_Insured_premium'] = data.groupby(['Previously_Insured'])['log_premium'].transform('mean')
    return data

params=['Gender', 'Age_bin', 'Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium']

def preprocessing(values):
    data = pd.DataFrame(values.reshape(1,7),columns=params)
    age =data['Age_bin']
    data.drop('Age_bin',axis=1,inplace=True)
    data['log_premium'] = data['Annual_Premium'].apply(lambda x: np.log(x))
    data.drop('Annual_Premium',axis=1,inplace=True)
    data['Gender_premium'] = data.groupby(['Gender'])['log_premium'].transform('mean')
    data['Vehicle_Damage_premium'] = data.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    data['Vehicle_Age_premium'] = data.groupby(['Vehicle_Age'])['log_premium'].transform('mean')
    data['Previously_Insured_premium'] = data.groupby(['Previously_Insured'])['log_premium'].transform('mean')
    df = pd.concat([data,age],axis=1)
    for i in params:
        if i not in ['Annual_Premium']:
            df[i] = df[i].astype('category')
    return df