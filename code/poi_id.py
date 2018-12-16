#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import matplotlib.pyplot
import pprint
pp = pprint.PrettyPrinter(indent=4)
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree,metrics,cross_validation
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'salary', 'deferred_income', 'expenses'] # You will need to use more features
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print len(data_dict) #数据点总数为146
#共标注18个嫌疑人
poi_count = 0
for person_name in data_dict:
    if data_dict[person_name]["poi"] ==1:
        poi_count = poi_count + 1
print(poi_count)
#找出缺失值
def count_nan(dataset):
    d = {}
    for person in dataset:
        for key, value in dataset[person].iteritems():
            if value == "NaN":
                if key in d:
                    d[key] += 1
                else:
                    d[key] = 1
    return d
print "* List of NaNs per feature:"
pp.pprint(count_nan(data_dict))
### Task 2: Remove outliers
#找到异常值
max = 0
max_name=""
for i in data_dict:
    if (data_dict[i]["bonus"]>max and data_dict[i]["bonus"]!="NaN"):
        max=data_dict[i]["bonus"]
        max_name=i
print "异常值为",max_name
data_dict.pop( "TOTAL", 0 )#清除异常值
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
data=data_dict.pop("LOCKHART EUGENE E",0)
### Task 3: Create new feature(s)
def poi_email_ratio(from_poi_to_this_person, to_messages):
    if from_poi_to_this_person or to_messages == 'NaN':
        to_poi_ratio = 0
    else:
        to_poi_ratio = float(from_poi_to_this_person)/to_messages
    return to_poi_ratio
#绘制可视化图，预估新特征对算法的影响
for employee, persons in data_dict.iteritems():
        if persons['from_this_person_to_poi'] == 'NaN' or persons['from_messages'] == 'NaN':
            persons['to_poi_ratio'] = 'NaN'
        else:
            persons['to_poi_ratio'] = float(persons['from_this_person_to_poi']) / float(persons['from_messages'])
            
        if persons['from_poi_to_this_person'] == 'NaN' or persons['to_messages'] == 'NaN':
            persons['from_poi_ratio'] = 'NaN'
        else:
            persons['from_poi_ratio'] = float(persons['from_poi_to_this_person']) / float(persons['to_messages'])
        
features = ["to_poi_ratio","from_poi_ratio","poi"]
data = featureFormat(data_dict, features)
for point in data:
    to_poi_ratio = point[0]
    from_poi_ratio = point[1]
    poi=point[2]
    if poi==1:
        matplotlib.pyplot.scatter( to_poi_ratio, from_poi_ratio, c='r')
    if poi==0:
        matplotlib.pyplot.scatter( to_poi_ratio, from_poi_ratio, c='b' )
matplotlib.pyplot.xlabel("to_poi_ratio")
matplotlib.pyplot.ylabel("from_poi_ratio")
matplotlib.pyplot.show()
### Store to my_dataset for easy export below.算法性能影响测试
my_dataset = data_dict
for key in my_dataset:
    my_dataset[key]['to_poi_ratio'] = poi_email_ratio(my_dataset[key]['from_poi_to_this_person'],  my_dataset[key]['to_messages'])
#改变features_list，分别包含和不包含新特征，用tester.py报告recall和precision,测试新特征对性能的影响
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
label, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, label, test_size=0.1)

clf_addnew = tree.DecisionTreeClassifier()
clf_addnew = clf_addnew.fit(features_train, labels_train)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
#直接预测
#朴素贝叶斯预测
clf_bay = GaussianNB()
clf_bay.fit(features_train, labels_train)

#决策树预测
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(features_train, labels_train)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#自动调参   决策树
clf=DecisionTreeClassifier()
parameters = {'criterion':('gini','entropy'),'max_depth': [None,2,5,10],'min_samples_leaf':[1,5,10],'max_leaf_nodes':[None,5,10,20],'min_samples_split':[2,10,20]}
clf=grid=GridSearchCV(clf,parameters,scoring='f1')
clf=grid.fit(features_train, labels_train)
print "best estimator:",clf.best_estimator_
print "best score:", clf.best_score_
clf=clf.best_estimator_
#用最优参数预测
clf = clf.fit(features_train, labels_train)
print data_dict
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)