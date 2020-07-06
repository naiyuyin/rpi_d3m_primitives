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
#from common_primitives import dataset_remove_columns
#from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import time

#from rpi_d3m_primitives.JMIplus import JMIplus
from rpi_d3m_primitives.JMIplus_auto import JMIplus_auto
from rpi_d3m_primitives.STMBplus_auto import STMBplus_auto
from rpi_d3m_primitives.S2TMBplus import S2TMBplus

import d3m.primitives.data_cleaning.imputer as Imputer
import d3m.primitives.classification.random_forest as RF
import d3m.primitives.classification.bagging as Bagging
import d3m.primitives.classification.gradient_boosting as GB 
import d3m.primitives.classification.extra_trees as ET
from rpi_d3m_primitives.TreeAugmentedNB_BayesianInf import TreeAugmentedNB_BayesianInf as TAN_BAY
import d3m.primitives.data_preprocessing.robust_scaler as Robustscaler
import d3m.primitives.data_preprocessing.min_max_scaler as MMscaler #SKlearn
from common_primitives.extract_columns import ExtractColumnsPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.remove_semantic_types import RemoveSemanticTypesPrimitive


# dataset_name = '38_sick' # target_index = 30 metric= f1 posLabel= sick 
# dataset_name = '57_hypothyroid' #target = 30 metric = f1 macro SCORE/dataset_TEST
# dataset_name = '27_wordLevels' #target = 13 metric = f1 macro SCORE/dataset_TEST
# dataset_name = '313_spectrometer' # target = 2 metric = f1 macro SCORE/dataset_TEST remove col 1 JMI-counting
# dataset_name = 'LL0_1100_popularkids' #target = 7 metric = f1 macro SCORE/dataset_TEST JMI-counting
# dataset_name = '1491_one_hundred_plants_margin' # target = 65, metric = f1 macro SCORE/dataset_TEST
#dataset_name = 'LL0_186_braziltourism' 	#target = 9 metric = f1 macro SCORE/dataset_SCORE 	
# dataset_name = 'LL0_acled_reduced' # target = 6  metric = accuracy SCORE/dataset_TEST		JMI-pseudoBayesian -> remove col 7 8 10 13
# dataset_name = '299_libras_move' #target = 91 metric = 	accuracy SCORE/dataset_TEST		
# dataset_name = 'LL1_336_MS_Geolife_transport_mode_prediction' #target = 7 metric = accuracy SCORE/dataset_SCORE remove col 1,4		
# dataset_name = '1567_poker_hand' #target = 11 metric = f1_macro  SCORE/dataset_TEST 
# dataset_name = '185_baseball' #target = 18 metric = f1 macro 	SCORE/dataset_TEST 


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
dataset_name = '185_baseball_MIN_METADATA'  #target = 18, metric = f1 macro  SCORE/dataset_SCORE


target_index = 18
score_file_name = "dataset_SCORE"
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
filename = dataset_name + '-tan.txt'

fs_nbins_lower_bound = 4
fs_nbins_upper_bound = 5

nbins_lower_bound, nbins_upper_bound, N0_lower_bound, N0_upper_bound = 7, 8, 2, 3
feal_list = ['non']
method = {'STMB':['pseudoBayesian'], 'S2TMB':[], 'JMI': ['counting'], 'non':[]}


print('\ndataset to dataframe')   
# step 1: dataset to dataframe
# path = os.path.join('/home/naiyu/Desktop/D3M_seed_datasets', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

#==============================training dataset================================
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 15), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 10), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 13), 'https://metadata.datadrivendiscovery.org/types/Attribute')



print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\n metadata generation')
hyperparams_class = SimpleProfilerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
profile_primitive = SimpleProfilerPrimitive(hyperparams=hyperparams_class.defaults().replace({'detect_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute','https://metadata.datadrivendiscovery.org/types/PrimaryKey']}))
profile_primitive.set_training_data(inputs = dataframe)
profile_primitive.fit()
call_metadata = profile_primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\n remove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
hyperparams_class = RemoveSemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = RemoveSemanticTypesPrimitive(hyperparams = hyperparams_class.defaults().replace({'columns': [target_index, 1], 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\nColumn Parser')
hyperparams_class = column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
dataframe = primitive.produce(inputs=dataframe).value

print('\nExtract Attributes')
# hyperparams_class = ExtractColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = ExtractColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({'columns': att_columns}))
# call_metadata = primitive.produce(inputs=dataframe)
# trainD = call_metadata.value
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
trainD = call_metadata.value

# print('\nImpute trainD')
# hyperparams_class = Imputer.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# Imputer_primitive = Imputer.SKlearn(hyperparams=hyperparams_class.defaults().replace({'strategy':'most_frequent'}))
# Imputer_primitive.set_training_data(inputs=trainD)
# Imputer_primitive.fit()
# trainD = Imputer_primitive.produce(inputs=trainD).value

# print('\nRobust Scaler') 
# hyperparams_class = Robustscaler.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# scale_primitive = Robustscaler.SKlearn(hyperparams=hyperparams_class.defaults())
# scale_primitive.set_training_data(inputs=trainD)
# scale_primitive.fit() 
# trainD = scale_primitive.produce(inputs=trainD).value


print('\nExtract Targets')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/TrueTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
trainL = call_metadata.value

#==============================testing dataset=================================
print ('\nLoad testing dataset') 
# path = os.path.join('/home/naiyu/Desktop/D3M_seed_datasets/', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
path = os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_tpye(('learningData', metadata_base.ALL_ELEMENTS, 15), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 10), 'https://metadata.datadrivendiscovery.org/types/Attribute')
# dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 13), 'https://metadata.datadrivendiscovery.org/types/Attribute')



print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\n metadata generation')
call_metadata = profile_primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\n remove semantic type')
# dataframe.metadata = dataframe.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')
hyperparams_class = RemoveSemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = RemoveSemanticTypesPrimitive(hyperparams = hyperparams_class.defaults().replace({'columns': [target_index, 1], 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
dataframe = call_metadata.value

print('\nColumn Parser')
hyperparams_class = column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
dataframe = primitive.produce(inputs=dataframe).value

print('\nExtract Attributes')
# hyperparams_class = ExtractColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
# primitive = ExtractColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({'columns': att_columns}))
# call_metadata = primitive.produce(inputs=dataframe)
# testD = call_metadata.value
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
testD = call_metadata.value

# print('\nImpute testD')
# testD = Imputer_primitive.produce(inputs=testD).value

# print('\nScale')
# testD = scale_primitive.produce(inputs=testD).value

print('\nExtract Suggested Target')
hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TrueTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
testL = call_metadata.value

print('\nGet Target Name')
column_metadata = testL.metadata.query((metadata_base.ALL_ELEMENTS, 0))
TargetName = column_metadata.get('name',[])






for f in feal_list:
	if len(method[f]) == 0:
		best_score = 0
		best_param = ""
		str_line = ""
		# with open(os.path.join('/home/naiyu/Desktop/D3M_seed_datasets/',dataset_name, file_name),"w+") as f_output:
		with open(os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current',dataset_name, filename),"w+") as f_output:  
			f_output.write('Method: ' + f + '\n')
			f_output.write("feat_sel_nbins \t feat_sel_idx \t feat_sel_num \t nbins \t classifier \t F1_score\n")
			trainD_org, trainL_org, testD_org, testL_org = trainD, trainL, testD, testL
			for fs_nbins in range(fs_nbins_lower_bound,fs_nbins_upper_bound,1): 
					str_fnbins =str_line +  str(fs_nbins)+'\t'
					trainD_c, trainL_c, testD_c, testL_c = trainD_org, trainL_org, testD_org, testL_org
					if f == 'non':
						print('Oops! No Feature Selection.')
						str_feal = str_fnbins + 'ALL\t'
						str_num = str_feal + str(1) + '\t'
					elif f == 'S2TMB':
						print('S2TMB Feature Selection Initiated')
						hyperparams_class = S2TMBplus.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
						FSmodel = S2TMBplus(hyperparams=hyperparams_class.defaults().replace({'nbins':fs_nbins, 'strategy':'uniform'}))
						FSmodel.set_training_data(inputs=trainD_c, outputs=trainL_c)
						print('\nSelected Feature Index')
						print(FSmodel._index)
						print('\n')
						if FSmodel._index is not None and len(FSmodel._index) is not 0:
							trainD_c = FSmodel.produce(inputs=trainD_c)
							trainD_c = trainD_c.value
							print('\nSubset of testing data')
							testD_c = FSmodel.produce(inputs=testD_c)
							testD_c = testD_c.value
							str_feal = str_fnbins + str(FSmodel._index) + '\t'
							str_num = str_feal + str(len(FSmodel._index)) + '/' + str(np.shape(trainD_c)[1]) + '\t'
						else:
							str_feal = str_fnbins + 'ALL\t'
							str_num = str_feal + str(1) + '\t'

					for nbins in range(nbins_lower_bound,nbins_upper_bound,1):
						str_nbins = str_num + str(nbins) + '\t'
						str_class = str_nbins + 'TAN' + '\t'
						print('The nbins is %d\n'%nbins)
						for N0 in range(N0_lower_bound,N0_upper_bound,1):
							str_n0 = str_class +  str(N0) + '\t'
							hyperparams_class = TAN_BAY.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
							classifier = TAN_BAY(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins,'N0':N0, 'strategy': 'uniform'}))
							classifier.set_training_data(inputs=trainD_c, outputs=trainL_c)
							classifier.fit()
							predictedTargets = classifier.produce(inputs=testD_c)
							predictedTargets = predictedTargets.value

							print('\nConstruct Predictions')
							hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
							construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
							call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
							dataframe = call_metadata.value

							print('\ncompute scores')
							# path = os.path.join('/home/naiyu/Desktop/D3M_seed_datasets/', dataset_name, 'SCORE', score_file, 'datasetDoc.json')
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

							str_line_final = str_n0 +  str(scores.iat[0,1])+'\t\n'
							f_output.write(str_line_final)
							if scores.iat[0,1] > best_score:
								best_score = scores.iat[0,1]
								best_param = str_line_final

			f_output.write("the best\n")
			f_output.write(best_param)
			f_output.close()


	
	elif len(method[f]) != 0:			
		for m in method[f]:
			# with open(os.path.join('/home/naiyu/Desktop/D3M_seed_datasets/',dataset_name, file_name),"w+") as f_output:
			with open(os.path.join('/Users/naiyuyin/Desktop/datasets/seed_datasets_current',dataset_name, filename),"w+") as f_output:  
				best_score = 0
				best_param = ""
				str_line = ""
				f_output.write("Method: " + f + '-' + m + '\t\n')
				f_output.write("feat_sel_nbins \t feat_sel_idx \t feat_sel_num \t nbins \t classifier \t F1_score\n")
				trainD_org, trainL_org, testD_org, testL_org = trainD, trainL, testD, testL
				for fs_nbins in range(fs_nbins_lower_bound,fs_nbins_upper_bound,1): #fs_nbins is the nbins for feature selection
					str_fnbins =str_line + str(fs_nbins)+'\t'
					trainD_c, trainL_c, testD_c, testL_c = trainD_org, trainL_org, testD_org, testL_org
					if f == 'STMB':
						print('The STMB Feature Selection Method Initiated.')
						hyperparams_class = STMBplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
						FSmodel = STMBplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':fs_nbins, 'method': m, 'strategy':'uniform'}))
					elif f == 'JMI':
						print('The JMI Feature Selection Method Initiated.')
						hyperparams_class = JMIplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
						FSmodel = JMIplus_auto(hyperparams=hyperparams_class.defaults().replace({'nbins':fs_nbins, 'method': m}))
					FSmodel.set_training_data(inputs=trainD_c, outputs=trainL_c)        
					FSmodel.fit()
					print('\nSelected Feature Index')
					print(FSmodel._index)
					print(len(FSmodel._index))
					print('\n')
					# idx = []?
					if FSmodel._index is not None and len(FSmodel._index) is not 0:
						trainD_c = FSmodel.produce(inputs=trainD_c)
						trainD_c = trainD_c.value
						print('\nSubset of testing data')
						testD_c = FSmodel.produce(inputs=testD_c)
						testD_c = testD_c.value
						str_feal = str_fnbins + str(FSmodel._index) + '\t'
						str_num = str_feal + str(len(FSmodel._index)) + '/' + str(np.shape(trainD)[1])+ '\t'
					else:
						str_feal = str_fnbins + 'ALL\t'
						str_num = str_feal + str(1) + '\t'

#					print('\nImpute trainD')
#					hyperparams_class = Imputer.SKlearn.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#					Imputer_primitive = Imputer.SKlearn(hyperparams=hyperparams_class.defaults().replace({'strategy':'most_frequent'}))
#					Imputer_primitive.set_training_data(inputs=trainD_c)
#					Imputer_primitive.fit()
#					trainD_c = Imputer_primitive.produce(inputs=trainD_c).value
#					print('\nImpute testD')
#					testD_c = Imputer_primitive.produce(inputs=testD_c).value

					for nbins in range(nbins_lower_bound,nbins_upper_bound,1): #n_bins is for the TAN classifier
						print(nbins)
						str_nbins =str_num + str(nbins) + '\t'
						str_class =str_nbins +  'TAN' + '\t'
						for N0 in range(N0_lower_bound,N0_upper_bound,1):
							print(N0)
							str_n0 = str_class + str(N0) + '\t'
							hyperparams_class = TAN_BAY.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
							classifier = TAN_BAY(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins,'N0':N0, 'strategy': 'quantile'}))
							classifier.set_training_data(inputs=trainD_c, outputs=trainL_c)
							classifier.fit()
							predictedTargets = classifier.produce(inputs=testD_c)
							predictedTargets = predictedTargets.value

							print('\nConstruct Predictions')
							hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
							construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
							call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
							dataframe = call_metadata.value
                        
							print('\ncompute scores')
							# path = os.path.join('/home/naiyu/Desktop/D3M_seed_datasets/', dataset_name, 'SCORE',score_file,'datasetDoc.json')    
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
                        
							str_line_final = str_n0 +  str(scores.iat[0,1])+'\t\n'
							f_output.write(str_line_final)
							if scores.iat[0,1] > best_score:
								best_score = scores.iat[0,1]
								best_param = str_line_final
				f_output.write("the best\n")
				f_output.write(best_param)
				f_output.close()






		

