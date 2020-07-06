import os
import numpy as np
from d3m import container
from collections import OrderedDict
from d3m import container, utils
from common_primitives import utils as comUtils
from d3m.metadata import base as metadata_base
from d3m import metrics
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives import ndarray_to_dataframe
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.unseen_label_encoder import UnseenLabelEncoderPrimitive
from common_primitives.unseen_label_decoder import UnseenLabelDecoderPrimitive
from common_primitives import construct_predictions
from d3m.primitives.evaluation import compute_scores
from common_primitives import extract_columns_semantic_types, column_parser, utils
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import time
from rpi_d3m_primitives.S2TMBplus import S2TMBplus

from rpi_d3m_primitives.JMIplus_auto import JMIplus_auto
from rpi_d3m_primitives.STMBplus_auto import STMBplus_auto
from rpi_d3m_primitives.S2TMBplus import S2TMBplus
import d3m.primitives.data_cleaning.imputer as Imputer

import d3m.primitives.regression.gradient_boosting as GB 
import d3m.primitives.regression.random_forest as RF 
import d3m.primitives.regression.extra_trees as ET 
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.remove_semantic_types import RemoveSemanticTypesPrimitive
#### Phrase I datasets
# dataset_name = '26_radon_seed' #regression #target_index = 29
# dataset_name = '534_cps_85_wages' #target_index = 6
dataset_name = 'LL0_207_autoPrice' #regression #target_index = 16
# dataset_name = '196_autoMpg' #regression #target_index = 8

# dataset_name = '26_radon_seed' #regression #target_index = 29
# dataset_name = '534_cps_85_wages_MIN_METADATA' #target_index = 6
# dataset_name = 'LL0_207_autoPrice' #regression #target_index = 16
# dataset_name = '196_autoMpg_MIN_METADATA' #regression #target_index = 8

#### Phrase II datasets:
# dataset_name = 'DA_college_debt' # target = 18, metric = root mean squared error
# dataset_name = 'DA_medical_malpractice' # target = 9, metric = root mean squared error 
# dataset_name = 'DA_ny_taxi_demand' # target = 2, metric = mean absolute error
# dataset_name = 'DA_poverty_estimation' #target = 5, metric = mean absolute error


target_index = 16
f = 'STMB'
# nbins_lower_bound = 9
# nbins_upper_bound = 10
# n_estimators_lower_bound = 17
# n_estimators_upper_bound = 18
nbins_lower_bound, nbins_upper_bound, n_estimators_lower_bound, n_estimators_upper_bound = 8, 9, 20, 21
filename = dataset_name + '-' + f + '-test.txt'
# gridsearch
method_list = {'STMB': ['counting', 'BayesFactor', 'pseudoBayesian', 'fullBayesian'], 'S2TMB':['None'], 'JMI':['counting', 'pseudoBayesian', 'fullBayesian']}
method = method_list[f]
Classifiers = ['RF', 'GB', 'ET']
# verify
# method = ['pseudoBayesian']
# Classifiers = ['ET']
score_file_name = 'dataset_SCORE'

if dataset_name in ['26_radon_seed_MIN_METADATA', 'DA_college_debt', 'DA_medical_malpractice']:
    metric = 'ROOT_MEAN_SQUARED_ERROR'
elif dataset_name in ['534_cps_85_wages_MIN_METADATA', '196_autoMpg_MIN_METADATA', 'LL0_207_autoPrice_MIN_METADATA']:
    metric = 'MEAN_SQUARED_ERROR'
elif dataset_name in ['DA_ny_taxi_demand', 'DA_poverty_estimation']:
    metric = 'MEAN_ABSOLUTE_ERROR'



print('\ndataset to dataframe')   
# step 1: dataset to dataframe
path = os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
# path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

#==============================training dataset================================

#target_index = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))['dimension']['length']-1
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
#dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 15), 'https://metadata.datadrivendiscovery.org/types/Attribute')
#dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 19), 'https://metadata.datadrivendiscovery.org/types/Attribute')

##1**********************
#print('\nRemove Columns')
#remove_columns_hyperparams_class = dataset_remove_columns.RemoveColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#hp = remove_columns_hyperparams_class({'columns': [1], 'resource_id': 'learningData'})
#remove_columns_primitive = dataset_remove_columns.RemoveColumnsPrimitive(hyperparams=hp)
#dataset = remove_columns_primitive.produce(inputs=dataset).value
#**************************

print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

# print('\n metadata generation')
# hyperparams_class = SimpleProfilerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# profile_primitive = SimpleProfilerPrimitive(hyperparams=hyperparams_class.defaults().replace({'detect_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
#                 'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute','https://metadata.datadrivendiscovery.org/types/PrimaryKey']}))
# profile_primitive.set_training_data(inputs = dataframe)
# profile_primitive.fit()
# call_metadata = profile_primitive.produce(inputs=dataframe)
# dataframe = call_metadata.value

print('\n remove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
hyperparams_class = RemoveSemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = RemoveSemanticTypesPrimitive(hyperparams = hyperparams_class.defaults().replace({'columns': [target_index], 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\nColumn Parser')
hyperparams_class = column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
dataframe = primitive.produce(inputs=dataframe).value

print('\nExtract Attributes')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
trainD = call_metadata.value

print('\nExtract Targets')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/TrueTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
trainL = call_metadata.value

#==============================testing dataset=================================
print ('\nLoad testing dataset') 
path = os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets/', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
# path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
#dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 15), 'https://metadata.datadrivendiscovery.org/types/Attribute')
#dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 19), 'https://metadata.datadrivendiscovery.org/types/Attribute')

##2*************************
#print('\nRemove Column')
#dataset = remove_columns_primitive.produce(inputs=dataset).value
##***************************

print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

# print('\n metadata generation')
# call_metadata = profile_primitive.produce(inputs=dataframe)
# dataframe = call_metadata.value

print('\n remove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
hyperparams_class = RemoveSemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = RemoveSemanticTypesPrimitive(hyperparams = hyperparams_class.defaults().replace({'columns': [target_index], 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\nColumn Parser')
hyperparams_class = column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
dataframe = primitive.produce(inputs=dataframe).value

print('\nExtract Attributes')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
testD = call_metadata.value

print('\nExtract Suggested Target')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TrueTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
testL = call_metadata.value

print('\nGet Target Name')
column_metadata = testL.metadata.query((metadata_base.ALL_ELEMENTS, 0))
TargetName = column_metadata.get('name',[])

best_score = 1e9
best_param = ""
#prepare the output file
with open(os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets',dataset_name, filename),"w+") as f_output:  
# with open(os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current',dataset_name, filename),"w+") as f_output:
    f_output.write("nbins \n feat_sel \t method \t selected features \t num_trees \t classifiers \t F1_score\n")
    str_line = ""#the output string for each line 
    trainD_org = trainD
    trainL_org = trainL
    testD_org = testD
    testL_org = testL
    for nbins in range(nbins_lower_bound, nbins_upper_bound,1):#loop for nbins
        print("\nThe current nbins is %d"%nbins)
        str_nbins=str_line+str(nbins)+'\t'
        for m in method:
            print("\nThe current mutual information method is %s"%m)
            trainD = trainD_org
            testD = testD_org
            trainL = trainL_org
            testL = testL_org
            #==============print('\nFeature Selection: S2TMB')
            #step 6 feature selection
            str_f = str_nbins+ f +'\t'#loop for feature selection method
            str_m = str_f + m + '\t'
            if f == 'non':
                trainD = trainD_org
                testD = testD_org
                trainL = trainL_org
                testL = testL_org
                str_line_fea=str_m + 'all\t'
            else:           
                if f == 'STMB':
                    print("\nThe current feature selection method is STMB")
                    hyperparams_class = STMBplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    FSmodel = STMBplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins, 'method': m, 'problem_type':'regression'}))
                elif f == 'S2TMB':
                    print("\nThe current feature selection method is S2TMB")
                    hyperparams_class = S2TMBplus.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    FSmodel = S2TMBplus(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins}))                  
                else:#JMI
                    print("\nThe current feature selection method is JMI")
                    hyperparams_class = JMIplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    FSmodel = JMIplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins, 'method': m}))
                FSmodel.set_training_data(inputs=trainD, outputs=trainL)        
                FSmodel.fit()
                print('\nSelected Feature Index')
                print(FSmodel._index)
                print('\n')
                if not len(FSmodel._index) == 0:
                    trainD = FSmodel.produce(inputs=trainD) 
                    trainD = trainD.value
                    print('\nSubset of testing data')
                    testD = FSmodel.produce(inputs=testD)
                    testD = testD.value
                    str_line_fea=str_m + str(FSmodel._index) + '\t'#for output
            if not len(FSmodel._index) == 0:
                ##=================================================================
                print('\nImpute trainD')
                hyperparams_class = Imputer.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                Imputer_primitive = Imputer.SKlearn(hyperparams=hyperparams_class.defaults().replace({'strategy':'most_frequent'}))
                Imputer_primitive.set_training_data(inputs=trainD)
                Imputer_primitive.fit()
                trainD = Imputer_primitive.produce(inputs=trainD).value
                print('\nImpute testD')
                testD = Imputer_primitive.produce(inputs=testD).value
                ##=================================================================
                for n_estimators in range (n_estimators_lower_bound, n_estimators_upper_bound,1):      
                    print("\nThe current number of trees in classifiers is %d"%n_estimators)
                    str_nestimators = str_line_fea + str(n_estimators) + '\t'
                    #=========================classifiers with d3m sklearn========
                    for classifier in Classifiers:
                        str_classifier = str_nestimators + classifier + '\t'
                        if classifier == 'RF':
                            print("\nThe current regressor is %s"%classifier)
                            hyperparams_class = RF.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            RF_primitive = RF.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators}), random_seed = 7)
                            RF_primitive.set_training_data(inputs=trainD, outputs=trainL)
                            RF_primitive.fit()
                            predictedTargets = RF_primitive.produce(inputs=testD)
                            predictedTargets = predictedTargets.value
                        elif classifier == 'GB':
                            print("\nThe current regressor is %s"%classifier)
                            hyperparams_class = GB.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            GB_primitive = GB.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators, 'learning_rate': 10 / n_estimators}), random_seed = 7)
                            GB_primitive.set_training_data(inputs=trainD, outputs=trainL)
                            GB_primitive.fit()
                            predictedTargets = GB_primitive.produce(inputs=testD)
                            predictedTargets = predictedTargets.value
                        elif classifier == 'ET':
                            print("\nThe current regressor is %s"%classifier)
                            hyperparams_class = ET.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            ET_primitive = ET.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators}), random_seed = 7)
                            ET_primitive.set_training_data(inputs=trainD, outputs=trainL)
                            ET_primitive.fit()
                            predictedTargets = ET_primitive.produce(inputs=testD)
                            predictedTargets = predictedTargets.value
                    
                    
                        print('\nConstruct Predictions')
                        hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                        construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
                        call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
                        dataframe = call_metadata.value
                        
                        print('\ncompute scores')
                        # path = os.path.join('/home/naiyu/Desktop/D3M_seed_datasets', dataset_name, 'SCORE', score_file_name,'datasetDoc.json')  
                        path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name, 'SCORE', score_file_name,'datasetDoc.json')                             
                        dataset = container.Dataset.load('file://{uri}'.format(uri=path))
                                        
                        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
                        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                                        
                        hyperparams_class = compute_scores.Core.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                        metrics_class = hyperparams_class.configuration['metrics'].elements
                        primitive = compute_scores.Core(hyperparams=hyperparams_class.defaults().replace({
                                    'metrics': [metrics_class({
                                        'metric': metric,
                                        'pos_label': None,
                                        'k': None,
                                    })],
                                    'add_normalized_scores': False,
                                }))
                        scores = primitive.produce(inputs=dataframe, score_dataset=dataset).value
                        
                        str_line_final = str_classifier + str(scores.iat[0,1])+'\t\n'
                        print("\nWrite to the txt file")
                        f_output.write(str_line_final)
                        if scores.iat[0,1] < best_score:
                            best_score = scores.iat[0,1]
                            best_param = str_line_final
        
        ##                    #3*******************************the version for unexpected value in train data
#                groundtruth_path = os.path.join('/home/yuru/Documents/D3Mdatasets-phase1/', dataset_name, 'SCORE/targets.csv')
#                GT_label = pd.read_csv(groundtruth_path)
#                GT_label = container.ndarray(GT_label[TargetName])
#        #        y_pred = predictedTargets.iloc[:,0]
#                y_pred = [int(i) for i in y_pred]
##                scores = accuracy_score(GT_label, y_pred)
#                scores = f1_score(GT_label, y_pred, average='macro')                                
#                str_line_final = str_line_final + str(scores)+'\t\n'
#                f_output.write(str_line_final)
#                if scores > best_score:
#                    best_score = scores
#                    best_param = str_line_final
#                
##*********************************************
                       
##*********************************************************************
    f_output.write("the best\n")
    f_output.write(best_param)
    f_output.close()                            






#groundtruth_path = os.path.join('/Users/zijun/Dropbox/', dataset_name, 'SCORE/targets.csv')
#GT_label = pd.read_csv(groundtruth_path)
#GT_label = container.ndarray(GT_label[TargetName])
#y_pred = predictedTargets.iloc[:,0]
#y_pred = [int(i) for i in y_pred]
#scores = f1_score(GT_label, y_pred, average='macro')

#print(scores)


#print('\nSave file')
#os.mkdir('/output/predictions/e7239570-bb9d-464b-aa5b-a0f7be958dc0')
#output_path = os.path.join('/output','predictions','e7239570-bb9d-464b-aa5b-a0f7be958dc0','predictions.csv')
#with open(output_path, 'w') as outputFile:
#    dataframe.to_csv(outputFile, index=False,columns=['d3mIndex', TargetName])
