import numpy as np
import pandas as pd
import re as re
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib as plt
import pygpu
from xgboost.sklearn import XGBClassifier
import SklearnHelper
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
#import theano

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

age_dist = pd.read_csv('data/age_gender_bkts.csv', header = 0)
users_train = pd.read_csv('data/train_users_2.csv', header = 0)
users_test = pd.read_csv('data/test_users.csv', header = 0)
train_count = users_train.shape[0]
#labels = users_train['country_destination'].values
id_test=users_test['id']
users = pd.concat((users_train,users_test),axis=0,ignore_index=True)

def feature_eng(users):
    us_age =age_dist[age_dist['country_destination']=='US']
    age_groups = us_age['age_bucket']

    age_groups = age_groups.str.extract('(\d+(-?)(\d+))',expand=True).iloc[:,0].str.split('-',expand=True)
    age_groups = age_groups.fillna(104).astype(int)
    age_groups= age_groups.values.tolist()
    us_age['age_bucket'] = age_groups

    us_age = us_age[us_age['age_bucket'].map(sum) > 24]
    us_age_male= us_age[us_age['gender']=='male']
    us_age_female= us_age[us_age['gender']=='female']

    agesGr = us_age_male['age_bucket'].values.tolist()
    agesGr.sort(key=lambda x: sum(x))
    agesDict = {v: k for v, k in enumerate(agesGr)}

    def findAgeGrpoup(age):
        for k, v in agesDict.items():
            ageFrom= v[0]
            ageTo = v[1]
            if ageFrom <= age <= ageTo:
                return k
        
    def findAgeGroupByArr(ageArr):
        for k, v in agesDict.items():
            if set(ageArr)==set(v):
                return k

    #GENDER DISTRIBUTION
    genders = ['MALE','FEMALE']
    gender_male_cnt = us_age[us_age['gender']=='male']['population_in_thousands'].sum()
    gender_female_cnt = us_age[us_age['gender']=='female']['population_in_thousands'].sum()
    gender_total = gender_male_cnt+gender_female_cnt
    genders_dist = [gender_male_cnt/gender_total,gender_female_cnt/gender_total]

    custm_gend = stats.rv_discrete(name='custm', values=([0,1], genders_dist))

    #male age distribution
    us_age_male['categ_age'] = us_age_male['age_bucket'].map(findAgeGroupByArr)
    us_age_male['pro']=us_age_male['population_in_thousands']/us_age_male['population_in_thousands'].sum()
    age_m_cat = us_age_male["categ_age"].tolist()
    age_m_dist = us_age_male['pro'].tolist()

    custm_male = stats.rv_discrete(name='custm', values=(age_m_cat, age_m_dist))

    #female age distribution
    us_age_female['categ_age'] = us_age_female['age_bucket'].map(findAgeGroupByArr)
    us_age_female['pro']=us_age_female['population_in_thousands']/us_age_female['population_in_thousands'].sum()
    age_f_cat = us_age_female["categ_age"].tolist()
    age_f_dist = us_age_female['pro'].tolist()

    custm_female = stats.rv_discrete(name='custm', values=(age_f_cat, age_f_dist))

    #FILL Gender unknown&OTHER
    users['gender'].replace('-unknown-', np.nan, inplace=True)
    users['gender'].replace('OTHER', np.nan, inplace=True)
    gender_null_count = users['gender'].isnull().sum()
    gender_rnd = custm_gend.rvs(size=gender_null_count)
    gender_val_rnd = list(map(lambda x: genders[x],gender_rnd ))
    users.loc[users['gender'].isnull(),'gender'] = gender_val_rnd

    #Fill age null
    users['categ_age'] = users['age'].map(findAgeGrpoup)
    #age_null_count = users['age'].isnull().sum()

    #fill male age
    age_male_null_count = users.loc[((users['categ_age'].isnull()) & (users['gender']=='MALE')),'categ_age'].shape[0]
    male_age_rand = custm_male.rvs(size=age_male_null_count)
    users.loc[((users['categ_age'].isnull()) & (users['gender']=='MALE')),'categ_age'] = male_age_rand

    #fill female age
    age_female_null_count = users.loc[((users['categ_age'].isnull()) & (users['gender']=='FEMALE')),'categ_age'].shape[0]
    female_age_rand = custm_male.rvs(size=age_female_null_count)
    users.loc[((users['categ_age'].isnull()) & (users['gender']=='FEMALE')),'categ_age'] = female_age_rand

    #split datetime
    users['date_account_created'] = pd.to_datetime(users['date_account_created'])
    users['dac_year'] = (users['date_account_created']).dt.year
    users['dac_month'] = (users['date_account_created']).dt.month
    users['dac_day'] = (users['date_account_created']).dt.day

    users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
    users['dfa_year'] = (users['date_first_active']).dt.year
    users['dfa_month'] = (users['date_first_active']).dt.month
    users['dfa_day'] = (users['date_first_active']).dt.day

    #Drop feat
    users.drop(['id','date_first_booking','age','date_account_created','date_first_active','country_destination'],inplace=True,axis=1,errors='ignore')

    #categories encoding features and drop them
    dum_feat=['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    for f in dum_feat:
        users_dummy = pd.get_dummies(users[f], prefix=f)
        users = users.drop([f], axis=1)
        users = pd.concat((users, users_dummy), axis=1)
    # for f in dum_feat:
    #     users[f] = users[f].astype('category')
    #     cat_feat = users[f].cat.codes
    #     users[f] = cat_feat
    return users

def first_level_model(x_train, y_train, x_test,y_test,enable_test=False):
    # Put in our parameters for said classifiers

    # Random Forest parameters criterion 'gini'
    rf_g_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        #  'max_depth': 6,
        #  'min_samples_leaf': 2,
        #  'max_features': 'sqrt',
        #  'verbose': 0
    }
    # Random Forest parameters criterion 'entropy'
    rf_e_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'criterion': 'entropy'
        #  'max_depth': 6,
        #  'min_samples_leaf': 2,
        #  'max_features': 'sqrt',
        #  'verbose': 0
    }
    # Extra Trees Parameters  criterion 'gini'
    et_g_params = {
        'n_jobs': -1,
         'n_estimators': 500,
        # # 'max_features': 0.5,
        # 'max_depth': 8,
        # 'min_samples_leaf': 2,
        # 'verbose': 0,
    }
    # Extra Trees Parameters criterion 'entropy'
    et_e_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'criterion': 'entropy'
        # # 'max_features': 0.5,
        # 'max_depth': 8,
        # 'min_samples_leaf': 2,
        # 'verbose': 0,
    }
    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample':0.5,
        'min_samples_leaf': 2,
    }

    SEED = 0
    # Create 5 objects that represent our 4 models
    rf_g = SklearnHelper.Helper(clf=RandomForestClassifier, seed=SEED, params=rf_g_params)
    rf_e = SklearnHelper.Helper(clf=RandomForestClassifier, seed=SEED, params=rf_e_params)
    et_g = SklearnHelper.Helper(clf=ExtraTreesClassifier, seed=SEED, params=et_g_params)
    et_e = SklearnHelper.Helper(clf=ExtraTreesClassifier, seed=SEED, params=et_e_params)
    gb = SklearnHelper.Helper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    num_class = np.unique(y_train).shape[0]
    clfs = [rf_g,rf_e,et_g,et_e,gb]
    x_tr = np.zeros((x_train.shape[0], len(clfs)*num_class))
    x_te = np.zeros((x_test.shape[0], len(clfs)*num_class))

    for j, clf in enumerate(clfs):
        oof_train, oof_test = SklearnHelper.Helper.get_oof(clf, x_train, y_train, x_test,y_test,enable_test)

        x_tr[:,j*num_class:j*num_class+num_class]= oof_train
        x_te[:,j*num_class:j*num_class+num_class] = oof_test
        #feature_import = clf.feature_importances(x_train, y_train)

    print("Training is complete")
    return x_tr,x_te

def second_level_model(x_train, y_train, x_test,y_test,enable_test=False):
    #TODO:CV & GridSeach
    xgb1 = XGBClassifier(
        max_depth=6,
        learning_rate=0.25,
        n_estimators=40,
        objective='multi:sdoftprob',
        subsample=0.6,
        colsample_bytree=0.6,
        seed=0
    )
    pred,proba = modelfit(xgb1,x_train, y_train)

    y_pred_x = xgb1.predict_proba(x_test)
    y_prd = xgb1.predict(x_test)
    if enable_test:
        acc_score = metrics.accuracy_score(y_test, y_prd)
        print("Test Accuracy : %.4g" % acc_score)
    return y_pred_x

def modelfit(alg, x_train,y_train,useTrainCV=False, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = np.unique(y_train).shape[0]
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(x_train, y_train,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:,1]
        
    #Print model report:
    print("\nModel Report")
    acc_score = metrics.accuracy_score(y_train, dtrain_predictions)
    print("CV Accuracy : %.4g" % acc_score)

    #feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    return dtrain_predictions,dtrain_predprob

def proceed_data(y_pred_x,id_tes):
    #Taking the 5 classes with highest probabilities
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(id_tes)):
        idx = id_tes[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_pred_x[i])[::-1])[:5].tolist()
    #Generate submission
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    # print(sub.info)
    return sub

def dataframe_to_csv(sub):
    print(sub)
    #spark_df = spark.createDataFrame(sub)
    #spark_df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/FileStore/repar/result")
    # display(dbutils.fs.ls("/FileStore/repar/result"))

train_count_lim=7000
test_cnt = 3000
labels = users[:train_count]['country_destination'].values#users[:train_count_lim]['country_destination'].values
labels_test = users[train_count_lim:train_count_lim+test_cnt]['country_destination'].values
#labels = users[:train_count]['country_destination'].values
id_test_f=users[train_count:]['id']#users[train_count:train_count_lim+train_count]['id']
#id_test_f=users[train_count:]['id']
users = feature_eng(users)
user_train_f = users[:train_count]
users_test_f = users[train_count:]
#users_train_lim = users[:train_count_lim]
values = users.values

X = user_train_f.values#[:train_count_lim]
le = LabelEncoder()
y = le.fit_transform(labels)
#X_test = values[train_count:]
#X_test = users_test_f.values[:train_count_lim]
X_test = user_train_f.values[train_count_lim:train_count_lim+test_cnt]
y_test = le.fit_transform(labels_test)


#x_tr,x_tes = first_level_model(X,y,X_test,y_test,True)
#pred = second_level_model(x_tr,y,x_tes,y_test,True)
pred =  second_level_model(X,y,X_test,y_test,True)
sub = proceed_data(pred,id_test_f.values)
sub.to_csv('sub.csv',index=False)
print('done')

