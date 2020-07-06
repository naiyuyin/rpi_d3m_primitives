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
from common_primitives import extract_columns_semantic_types, column_parser, utils, extract_columns
from common_primitives.cast_to_type import CastToTypePrimitive
#from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import time

from rpi_d3m_primitives.JMIplus_auto import JMIplus_auto
from rpi_d3m_primitives.STMBplus_auto import STMBplus_auto
from rpi_d3m_primitives.S2TMBplus import S2TMBplus
import d3m.primitives.data_cleaning.imputer as Imputer
import d3m.primitives.classification.random_forest as RF
import d3m.primitives.classification.bagging as Bagging
import d3m.primitives.classification.gradient_boosting as GB 
import d3m.primitives.classification.extra_trees as ET
import d3m.primitives.classification.xgboost_dart as XG
import d3m.primitives.data_preprocessing.min_max_scaler as MMscaler #SKlearn
import d3m.primitives.data_preprocessing.robust_scaler as Robustscaler #SKlearn
from common_primitives.denormalize import DenormalizePrimitive
import d3m.primitives.data_preprocessing.min_max_scaler as MMscaler #SKlearn
import d3m.primitives.data_preprocessing.robust_scaler as Robustscaler #SKlearn
# import d3m.primitives.schema_discovery.profiler as profiler
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.remove_semantic_types import RemoveSemanticTypesPrimitive


import timeit


#### Phrase I datasets:
# dataset_name = '38_sick' # target_index = 30 metric= f1 posLabel= sick  SCORE/dataset_SCORE
# dataset_name = '57_hypothyroid' #target = 30 metric = f1 macro SCORE/dataset_TEST
# dataset_name = '27_wordLevels' #target = 13 metric = f1 macro SCORE/dataset_TEST
# dataset_name = '313_spectrometer' # target = 2 metric = f1 macro SCORE/dataset_TEST remove col 1 JMI-counting
# dataset_name = 'LL0_1100_popularkids' #target = 7 metric = f1 macro SCORE/dataset_TEST JMI-counting
# dataset_name = '1491_one_hundred_plants_margin' # target = 65, metric = f1 macro SCORE/dataset_TEST
# dataset_name = 'LL0_186_braziltourism'    #target = 9 metric = f1 macro SCORE/dataset_SCORE   
# dataset_name = 'LL0_acled_reduced' # target = 6  metric = accuracy SCORE/dataset_TEST     JMI-pseudoBayesian -> remove col 7 8 10 13
# dataset_name = '299_libras_move' #target = 91 metric =    accuracy SCORE/dataset_SCORE    
# dataset_name = 'LL1_336_MS_Geolife_transport_mode_prediction' #target = 7 metric = accuracy SCORE/dataset_SCORE remove col 1,4        
# dataset_name = '1567_poker_hand' #target = 11 metric = f1_macro SCORE/dataset_TEST
dataset_name = '185_baseball'  #target = 18, metric = f1 macro  SCORE/dataset_TEST 


# dataset_name = '38_sick_MIN_METADATA' # target_index = 30 nbins, n_estimator = 9, 10, 27, 28 pseudo bagging 0.003125
# dataset_name = '57_hypothyroid_MIN_METADATA' #target = 30 metric = f1 macro SCORE/dataset_TEST
# dataset_name = '27_wordLevels_MIN_METADATA' #target = 13 metric = f1 macro SCORE/dataset_TEST
# dataset_name = '313_spectrometer_MIN_METADATA' # target = 2 metric = f1 macro SCORE/dataset_TEST remove col 1 JMI-counting
# dataset_name = 'LL0_1100_popularkids_MIN_METADATA' #target = 7 metric = f1 macro SCORE/dataset_TEST JMI-counting
# dataset_name = '1491_one_hundred_plants_margin_MIN_METADATA' # target = 65, metric = f1 macro SCORE/dataset_TEST
# dataset_name = 'LL0_186_braziltourism_MIN_METADATA'    #target = 9 metric = f1 macro SCORE/dataset_SCORE   
# dataset_name = 'LL0_acled_reduced_MIN_METADATA' # target = 6  metric = accuracy SCORE/dataset_TEST     JMI-pseudoBayesian -> remove col 7 8 10 13
# dataset_name = '299_libras_move_MIN_METADATA' #target = 91 metric =    accuracy SCORE/dataset_SCORE    
# dataset_name = 'LL1_336_MS_Geolife_transport_mode_prediction_MIN_METADATA' #target = 7 metric = accuracy SCORE/dataset_SCORE remove col 1,4        
# dataset_name = '1567_poker_hand_MIN_METADATA' #target = 11 metric = f1_macro SCORE/dataset_TEST
# dataset_name = '185_baseball_MIN_METADATA'  #target = 18, metric = f1 macro  SCORE/dataset_SCORE

#### Phrase I additional datasets:
# dataset_name = 'uu4_SPECT' # target = 1, priviledge = 46,47,48,49,50,51,52, 53, 54,55, 56,57, 58,69, 60, 61, 62, 63, 64, 65, 66,67, metric = f1, poslabel = 1
# dataset_name = 'uu5_heartstatlog' # target = 1, priviledge = 8, 9, 10, 11, 12, 13, 14, metric = f1, poslabel = 1
# dataset_name = 'uu6_hepatitis' # target = 1, priviledge = 20, metric = f1, poslabel = 1
# dataset_name = 'uu7_pima_diabetes' # target = 1, priviledge = 2, 8, metric = f1, poslabel = 1
# dataset_name = '4550_MiceProtein' # target = 82, metric = f1 macro SCORE/dataset_SCORE
# dataset_name = 'LL1_multilearn_emotions' # target = 73, metric = accuracy macro SCORE/dataset_SCORE

#### Phrase II datasets
# classification
# dataset_name = 'DA_fifa2018_manofmatch' # target = 14 metric = f1, poslabel = 1
# dataset_name = 'DA_consumer_complaints' # target = 7 metric = f1 macro 
#dataset_name = 'DA_housing_burden' # target = 287, metric = f1 macro
# dataset_name = 'DA_global_terrorism' # target = 4, metric = accuracy




target_index = 18
f = 'STMB'
# nbins_lower_bound = 2
# nbins_upper_bound = 3
# n_estimators_lower_bound = 2
# n_estimators_upper_bound = 3
nbins_lower_bound, nbins_upper_bound, n_estimators_lower_bound, n_estimators_upper_bound =10, 11, 14, 15
RS = 7 # random_seed
thres = 0.0375
filename = dataset_name + '-' + f + '-test.txt'
# gridsearch
# method_list = {'STMB': ['counting', 'BayesFactor', 'pseudoBayesian', 'fullBayesian'], 'S2TMB':['None'], 'JMI':['counting', 'pseudoBayesian', 'fullBayesian']}
# method = method_list[f]
# Classifiers = ['RF', 'Bagging', 'GB', 'ET', 'XG']
# verify
method = ['pseudoBayesian']
Classifiers = ['ET']
score_file_name = 'dataset_SCORE'
if dataset_name in ['38_sick','DA_fifa2018_manofmatch', 'uu4_SPECT', 'uu5_heartstatlog', 'uu6_hepatitis', 'uu7_pima_diabetes']:
    metric = 'F1'
elif dataset_name in ['299_libras_move','LL0_acled_reduced','LL1_336_MS_Geolife_transport_mode_prediction', 'LL1_multilearn_emotions',  'DA_global_terrorism']:
    metric = 'ACCURACY'
else:
    metric = 'F1_MACRO'
if dataset_name == "38_sick":
    poslabel = 'sick'
elif dataset_name in ["DA_fifa2018_manofmatch", 'uu4_SPECT', 'uu5_heartstatlog', 'uu6_hepatitis', 'uu7_pima_diabetes']:
    poslabel = '1'
else:
    poslabel = None


print('\ndataset to dataframe')   
# step 1: dataset to dataframe
path = os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
# path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

#print('\nDenormalization')
#hyperparams_class = DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#primitive = DenormalizePrimitive(hyperparams=hyperparams_class.defaults())
#call_metadata = primitive.produce(inputs=dataset)
#dataset = call_metadata.value



#==============================training dataset================================

# for i in range(1, 19, 1):
# 	dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, i), 'https://metadata.datadrivendiscovery.org/types/Attribute')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')

# print('\nRemove test data col 8, 9, 10, 11, 12, 13, 14 for uu5_heartstatlog')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 8), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 9), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 10), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 11), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 12), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 13), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 14), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# print('\nRemove test data col 2, 8 for uu7_pima_diabetes')
# # 185
# for i in range(1, 18, 1):
# 	dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, i), 'https://metadata.datadrivendiscovery.org/types/Attribute')

print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

# print('\nMetadata generation')
# hyperparams_class = SimpleProfilerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# profile_primitive = SimpleProfilerPrimitive(hyperparams=hyperparams_class.defaults().replace({'detect_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
#                 'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute','https://metadata.datadrivendiscovery.org/types/PrimaryKey']}))
# profile_primitive.set_training_data(inputs = dataframe)
# profile_primitive.fit()
# call_metadata = profile_primitive.produce(inputs=dataframe)
# dataframe = call_metadata.value

print('\nRemove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
hyperparams_class = RemoveSemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = RemoveSemanticTypesPrimitive(hyperparams = hyperparams_class.defaults().replace({'columns': [target_index], 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\nColumn Parser')
hyperparams_class = column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
dataframe = primitive.produce(inputs=dataframe).value

print('\nExtract Attributes by semantic types')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
trainD = call_metadata.value

# print('\nExtract Attributes by column index')
# hyperparams_class = extract_columns.ExtractColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = extract_columns.ExtractColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({'columns': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] }))
# call_metadata = primitive.produce(inputs=dataframe)
# trainD = call_metadata.value

#print('\nCast to Type')
#hyperparams_class = CastToTypePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#primitive = CastToTypePrimitive(hyperparams= hyperparams_class.defaults().replace({'type_to_cast': 'float'}))
#call_metadata = primitive.produce(inputs = trainD)
#trainD = call_metadata.value

# print('\nRobust Scaler')
# hyperparams_class = Robustscaler.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# scaler_primitive = Robustscaler.SKlearn(hyperparams=hyperparams_class.defaults())
# scaler_primitive.set_training_data(inputs=trainD)
# scaler_primitive.fit()
# trainD = scaler_primitive.produce(inputs=trainD).value

# print('\nMin Max Scaler')
# hyperparams_class = MMscaler.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# MM_primitive = MMscaler.SKlearn(hyperparams=hyperparams_class.defaults())
# MM_primitive.set_training_data(inputs=trainD)
# MM_primitive.fit()
# trainD = MM_primitive.produce(inputs=trainD).value

print('\nExtract Targets by semantic types')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/TrueTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
trainL = call_metadata.value

# print('\nExtract Targets by column index')
# hyperparams_class = extract_columns.ExtractColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = extract_columns.ExtractColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({'columns': [target_index] }))
# call_metadata = primitive.produce(inputs=dataframe)
# trainL = call_metadata.value

#==============================testing dataset=================================
print ('\nLoad testing dataset') 
path = os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
# path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

# print('\nDenormalization')
# hyperparams_class = DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = DenormalizePrimitive(hyperparams=hyperparams_class.defaults())
# call_metadata = primitive.produce(inputs=dataset)
# dataset = call_metadata.value

# for i in range(1, 19, 1):
# 	dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, i), 'https://metadata.datadrivendiscovery.org/types/Attribute')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')

# print('\nRemove test data col 8, 9, 10, 11, 12, 13, 14 for uu5_heartstatlog')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 8), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 9), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 10), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 11), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 12), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 13), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 14), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# print('\nRemove test data col 2, 8 for uu7_pima_diabetes')
## 185


print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

# print('\nMetadata generation')
# call_metadata = profile_primitive.produce(inputs=dataframe)
# dataframe = call_metadata.value

print('\nRemove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
hyperparams_class = RemoveSemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = RemoveSemanticTypesPrimitive(hyperparams = hyperparams_class.defaults().replace({'columns': [target_index], 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\nColumn Parser')
hyperparams_class = column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
dataframe = primitive.produce(inputs=dataframe).value

# print('\n remove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')

print('\nExtract Attributes by semantic types')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
testD = call_metadata.value

# print('\nExtract Attributes by column index')
# hyperparams_class = extract_columns.ExtractColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = extract_columns.ExtractColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({'columns': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] }))
# call_metadata = primitive.produce(inputs=dataframe)
# testD = call_metadata.value

#print('\nCast to Type')
#hyperparams_class = CastToTypePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#primitive = CastToTypePrimitive(hyperparams= hyperparams_class.defaults().replace({'type_to_cast': 'float'}))
#call_metadata = primitive.produce(inputs = testD)
#testD = call_metadata.value

# print('\nScaler')
# testD = scaler_primitive.produce(inputs=testD).value
# testD = MM_primitive.produce(inputs=testD).value

print('\nExtract Suggested Target by semantic types')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TrueTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
testL = call_metadata.value

# print("\nExtract Suggested Target by column index")
# hyperparams_class = extract_columns.ExtractColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = extract_columns.ExtractColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({'columns': [target_index] }))
# call_metadata = primitive.produce(inputs=dataframe)
# testL = call_metadata.value

print('\nGet Target Name')
column_metadata = testL.metadata.query((metadata_base.ALL_ELEMENTS, 0))
TargetName = column_metadata.get('name',[])

best_score = 0
best_param = ""

#prepare the output file
with open(os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets',dataset_name, filename),"w+") as f_output:  
# with open(os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current',dataset_name, filename),"w+") as f_output:  
    f_output.write("nbins\tfeat_sel\tmethod\tfs index\tnum_trees\tclassifier\tF1_score\n")
    str_line = ""#the output string for each line 
    trainD_org = trainD
    trainL_org = trainL
    testD_org = testD
    testL_org = testL
    for nbins in range(nbins_lower_bound,nbins_upper_bound,1):#loop for nbins
        print("\nThe current nbins is %d"%nbins)
        str_nbins=str_line+str(nbins)+'\t'
        # for f in range(len(list(fea_list))):#range(4):
        for m in method:
            print("\nThe current method of mutual information is %s"%m)
            trainD_c = trainD_org
            testD_c = testD_org
            trainL_c = trainL_org
            testL_c = testL_org
            str_f = str_nbins+ f +'\t'#loop for feature selection method
            str_m = str_f + m + '\t'
            # start  = timeit.default_timer()
            if f == 'non':
                trainD = trainD_org
                testD = testD_org
                trainL = trainL_org
                testL = testL_org
                str_line_fea=str_m + 'all\t'
            else:           
                if f == 'STMB':
                    print("\nGet into the STMB")
                    hyperparams_class = STMBplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    FSmodel = STMBplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins, 'method':m, 'thres_search_method': 'binary_search'}))
                    # FSmodel = STMBplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins, 'method':m}))
                elif f == 'S2TMB':
                    print("\nGet into the S2TMB")
                    hyperparams_class = S2TMBplus.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    FSmodel = S2TMBplus(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins}))                  
                else:#JMI
                    print("\nGet into the JMI")
                    hyperparams_class = JMIplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                    FSmodel = JMIplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins, 'method': m, 'strategy': 'quantile'}))
                FSmodel.set_training_data(inputs=trainD_c, outputs=trainL_c)        
                FSmodel.fit()
                print('\nSelected Feature Index')
                print(FSmodel._index)
                print('\n')
                if not len(FSmodel._index) == 0:
                    trainD_c = FSmodel.produce(inputs=trainD_c) 
                    trainD_c = trainD_c.value
                    print('\nSubset of testing data')
                    testD_c = FSmodel.produce(inputs=testD_c)
                    testD_c = testD_c.value
                    str_line_fea=str_m + str(FSmodel._index) + '\t'#for output
            # stop = timeit.default_timer()
            # print("The running time of one %s is %f\n"%(fea_list[f], stop-start))
            if not len(FSmodel._index) == 0:
                ##=================================================================
                print('\nImpute trainD')
                hyperparams_class = Imputer.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                Imputer_primitive = Imputer.SKlearn(hyperparams=hyperparams_class.defaults().replace({'strategy':'most_frequent'}))
                Imputer_primitive.set_training_data(inputs=trainD_c)
                Imputer_primitive.fit()
                trainD_c = Imputer_primitive.produce(inputs=trainD_c).value
                print('\nImpute testD')
                testD_c = Imputer_primitive.produce(inputs=testD_c).value
                
                # print('\nScaler')
                # hyperparams_class = Robustscaler.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                # scaler_primitive = Robustscaler.SKlearn(hyperparams=hyperparams_class.defaults())
                # scaler_primitive.set_training_data(inputs=trainD)
                # scaler_primitive.fit()
                # trainD = scaler_primitive.produce(inputs=trainD).value
                # testD = scaler_primitive.produce(inputs=testD).value
                ##=================================================================
                print('\nInitiate classification process')
                for n_estimators in range (n_estimators_lower_bound,n_estimators_upper_bound,1):  
                    print("\nThe current number of trees for classifier is %d"%n_estimators)    
                    str_nestimators = str_line_fea + str(n_estimators) + '\t'
                    #=========================classifiers with d3m sklearn========
                    for classifier in Classifiers:
                        print("\nThe current classifier is %s"%classifier)
                        str_classifier = str_nestimators + classifier + '\t'
                        if classifier == 'RF':
                            hyperparams_class = RF.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            RF_primitive = RF.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators}), random_seed = RS)
                            RF_primitive.set_training_data(inputs=trainD_c, outputs=trainL_c)
                            RF_primitive.fit()
                            predictedTargets = RF_primitive.produce(inputs=testD_c)
                            predictedTargets = predictedTargets.value
                        elif classifier == 'Bagging':
                            hyperparams_class = Bagging.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            Bagging_primitive = Bagging.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators}), random_seed = RS)
                            Bagging_primitive.set_training_data(inputs=trainD_c, outputs=trainL_c)
                            Bagging_primitive.fit()
                            predictedTargets = Bagging_primitive.produce(inputs=testD_c)
                            predictedTargets = predictedTargets.value
                        elif classifier == 'GB':
                            hyperparams_class = GB.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            GB_primitive = GB.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators, 'learning_rate': 10 / n_estimators}), random_seed = RS)
                            GB_primitive.set_training_data(inputs=trainD_c, outputs=trainL_c)
                            GB_primitive.fit()
                            predictedTargets = GB_primitive.produce(inputs=testD_c)
                            predictedTargets = predictedTargets.value
                        elif classifier == 'ET':
                            hyperparams_class = ET.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            # ET_primitive = ET.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators}), random_seed = RS)
                            ET_primitive = ET.SKlearn(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators, 'criterion': 'entropy'}), random_seed = RS)
                            ET_primitive.set_training_data(inputs=trainD_c, outputs=trainL_c)
                            ET_primitive.fit()
                            predictedTargets = ET_primitive.produce(inputs=testD_c)
                            predictedTargets = predictedTargets.value
                        elif classifier == 'XG':
                            hyperparams_class = XG.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                            XG_primitive = XG.Common(hyperparams=hyperparams_class.defaults().replace({'n_estimators':n_estimators, 'learning_rate': 1 / n_estimators}), random_seed = RS)
                            XG_primitive.set_training_data(inputs=trainD_c, outputs=trainL_c)
                            XG_primitive.fit()
                            predictedTargets = XG_primitive.produce(inputs=testD_c)
                            predictedTargets = predictedTargets.value                    
                        print('\nConstruct Predictions')
                        hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                        construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
                        call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
                        dataframe = call_metadata.value
                        
                        print('\ncompute scores')
                        # path = os.path.join('/Users/naiyuyin/Desktop/D3M_seed_datasets', dataset_name, 'SCORE', score_file_name,'datasetDoc.json')
                        path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name, 'SCORE', score_file_name,'datasetDoc.json')                               
                        dataset = container.Dataset.load('file://{uri}'.format(uri=path))
                                        
                        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
                        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
                                        
                        hyperparams_class = compute_scores.Core.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
                        metrics_class = hyperparams_class.configuration['metrics'].elements
                        primitive = compute_scores.Core(hyperparams=hyperparams_class.defaults().replace({
                                    'metrics': [metrics_class({
                                        'metric': metric,
                                        'pos_label': poslabel,
                                        'k': None,
                                    })],
                                    'add_normalized_scores': False,
                                }))
                        scores = primitive.produce(inputs=dataframe, score_dataset=dataset).value
                        # print("\nScores is %f"%scores.iat[0,1])
                        str_line_final = str_classifier + str(scores.iat[0,1])+'\t\n'
                        print("\nWrite the results to the txt!")
                        f_output.write(str_line_final)
                        if scores.iat[0,1] > best_score:
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
