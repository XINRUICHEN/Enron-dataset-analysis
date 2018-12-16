# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn import preprocessing
#生成数据集
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop( "TOTAL", 0 )
data_dict.pop("TRAVEL AGENCY IN THE PARK",0)
data=data_dict.pop("LOCKHART EUGENE E",0)
my_dataset=data_dict
#划分训练集与测试集
features_list = ['poi','exercised_stock_options','total_stock_value','bonus','deferral_payments','total_payments','salary','loan_advances','restricted_stock_deferred','deferred_income','expenses','long_term_incentive','restricted_stock','director_fees'] # You will need to use more features
data = featureFormat(my_dataset, features_list)
label, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, label, test_size=0.1, random_state=42)
#特征缩放
min_max_scaler = preprocessing.MinMaxScaler()
features_train=min_max_scaler.fit_transform(features_train)
features_test=min_max_scaler.fit_transform(features_test)
#选择特征
sfm = SelectKBest(f_classif, k=5).fit(features_train,labels_train)
selected_features=[]
for index,value in enumerate(sfm.get_support().flat):
    if value:
        selected_features.append(features_list[index + 1])
n_features = sfm.transform(features).shape[1]
print "the number of selected features:",n_features
print selected_features
scores = sfm.scores_
print scores


