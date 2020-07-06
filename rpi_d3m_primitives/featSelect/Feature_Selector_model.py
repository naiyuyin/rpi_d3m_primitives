# Algorithms
from rpi_d3m_primitives.featSelect.tian_STMB_new import tian_STMB_new
from rpi_d3m_primitives.featSelect.STMB_BayesFactor import STMB_BayesFactor as STMB_B_F
from rpi_d3m_primitives.featSelect.sSTMB import sSTMBplus
from rpi_d3m_primitives.featSelect.Large_Scale_STMB import Large_Scale_STMB
from rpi_d3m_primitives.featSelect.JointMutualInformation import jmi
from rpi_d3m_primitives.featSelect.tian_IPCMB import tian_IPCMB
from rpi_d3m_primitives.featSelect.Large_Scale_IPCMB import Large_Scale_IPCMB
from rpi_d3m_primitives.featSelect.Keyi_STMB_Adaptive_soft import Keyi_STMB_Adaptive_soft
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Classes
from rpi_d3m_primitives.featSelect.Predictor import Classifier,Regressor
# Packages
import numpy as np
import pickle
from pathlib import Path
import os


class FeatureSelector:
	def __init__(self,train_set, discrete_train_set, problem_type, indep_method = None, bayesfactors = None, test_set=None,predictor_base=None):
		self.train_set = train_set
		self.test_set = test_set
		self.discrete_train_set = discrete_train_set
		self.FEATURE_COUNT_THRESHOLD = 85
		self.problem_type = problem_type.lower()
		self.predictor_base = predictor_base
		self.set_predictor(predictor_base)
		self.indep_method = indep_method
		self.bayesfactors = bayesfactors
		if (self.problem_type not in ["regression","classification"]):
			raise ValueError("'problem_type' must be either 'regression' or 'classification'")

	def set_predictor(self,predictor_base):
		if (self.problem_type == "classification"):
			if (self.predictor_base == None):
#				self.predictor_base = AdaBoostClassifier(n_estimators=30)
				self.predictor_base = KNeighborsClassifier(n_neighbors=3)
			self.predictor = Classifier(self.predictor_base,self.test_set,self.train_set)
		elif (self.problem_type == "regression"):
			if (self.predictor_base == None):
				self.predictor_base = KNeighborsRegressor(n_neighbors=3)
			self.predictor = Regressor(self.predictor_base,self.test_set,self.train_set)

class MBFeatureSelector(FeatureSelector):
	def large_scale_selector(self,train_data,train_labels,thres):
		pass

	def small_scale_selector(self,train_data,train_labels,thres):
		pass

	def select_features(self, thres = None):
		# establish threshold for indepence
		 if (thres == None and self.indep_method != 'BayesFactor'):
		 	selected_features = self.find_optimal_threshold()[1]
		 	return selected_features
		 train_data = self.discrete_train_set.data
		 train_labels = self.discrete_train_set.labels
		 selected_features = None
		 if (self.discrete_train_set.num_features > self.FEATURE_COUNT_THRESHOLD and self.indep_method != 'BayesFactor'):
		 	selected_features = self.large_scale_selector(train_data,train_labels,thres)
		 else:
		 	selected_features = self.small_scale_selector(train_data,train_labels,thres)
		 return selected_features



	"""---------------THRESHOLD FINDING---------------"""
	def find_optimal_threshold(self,max_loop=8):
		return self.get_threshold(max_loop)

	def get_threshold(self,max_loop):
		pass

	def bisect(self,min_thres,max_thres,optimal_thres,optimal_feats,feats_to_accuracy,optimal_score):
		# Loop
		loop_count = 0
		max_loop = 3
		search_area = max_thres-min_thres
		left_thres = search_area/4+min_thres
		right_thres = 3*search_area/4+min_thres
		while (loop_count < max_loop):
			loop_count += 1
			# Choose
#			print('\nleft thres')
#			print(left_thres)
#			print('\nright thres')
#			print(right_thres)
			left_feats = self.select_features(left_thres)
			right_feats = self.select_features(right_thres)
			chosen_feats,chosen_score = \
				self.predictor.choose(left_feats,right_feats,optimal_feats,feats_to_accuracy)
#			print('\nchosen score')
#			print(chosen_score)
#			print('\noptimal score')
#			print(optimal_score)
			right_chosen = (np.array_equal(right_feats,chosen_feats))
			if chosen_score > optimal_score:
#			if (not (np.array_equal(chosen_feats, optimal_feats))):
				optimal_score = chosen_score
				optimal_feats = chosen_feats
				if (right_chosen):
					optimal_thres = right_thres
				else:
					optimal_thres = left_thres
			elif (chosen_score == optimal_score and len(chosen_feats) < len(optimal_feats)):
				optimal_score = chosen_score
				optimal_feats = chosen_feats
				if (right_chosen):
					optimal_thres = right_thres
				else:
					optimal_thres = left_thres
				# Divide
				#delta = (right_thres - left_thres) / 2
			delta = search_area/(2**(2+loop_count))
			if (right_chosen):
				left_thres = right_thres - delta
				right_thres = min(delta + right_thres, max_thres)
			else:
				right_thres = left_thres + delta
				left_thres = max(left_thres - delta, min_thres)
#			elif (list(left_feats) == [] and list(right_feats) == []): #search ares is too large
#				loop_count -= 1
#				search_area = search_area/2
#				left_thres = search_area/4 + min_thres
#				right_thres = 3*search_area/4 + min_thres
#			else:
#				# The increase in accuracy is insufficient to continue
#				break
			# Divide
			#delta = (right_thres-left_thres)/2
			#if (right_chosen):
			#	left_thres += delta
			#	right_thres = min(delta+right_thres,3)
			#else:
			#	left_thres = max(left_thres-delta,0)
			#	right_thres -= delta
		return optimal_thres,optimal_feats,optimal_score

	def get_threshold_bisection(self,max_loop):
		"""	Use bisection to find the optimal indepence threshold (used for STMB,IPCMB)
		"""
		"'-score' is a stand in for MSE when dealing with regression, and accuracy for classification"
		full_test_set = self.test_set
		num_samples = self.test_set.data.shape[0]
		# Constants
		MAX_THRESHOLD = 1
		MIN_THRESHOLD = 0
		Interv_THRESHOLD = 0.05
		num_Interv = (MAX_THRESHOLD-MIN_THRESHOLD)/Interv_THRESHOLD
		# Caches
		feats_to_accuracy = dict()
		# Initial optimal score will be computed with all features
		optimal_thres = 0
		optimal_feats = np.arange(0,self.train_set.data.shape[1])
		optimal_score = self.predictor.score(optimal_feats,feats_to_accuracy)
		for i in range(3):
			fold_start = int(i*num_samples/3)
			fold_end = int((i+1)*num_samples/3)
			self.predictor.test_set = full_test_set.resample(fold_start,fold_end)
			for min_thres in np.linspace(MIN_THRESHOLD,MAX_THRESHOLD,int(num_Interv+1)):
				optimal_thres, optimal_feats, optimal_score = self.bisect(min_thres, min_thres+Interv_THRESHOLD, optimal_thres,optimal_feats,feats_to_accuracy,optimal_score)
		optimal_feats = self.select_features(optimal_thres)
		self.predictor.test_set = full_test_set
		return optimal_thres,optimal_feats

class IPCMB(MBFeatureSelector):
	"Class that contains the algorithm for feature selection with IPCMB"
	def large_scale_selector(self,train_data,train_labels,thres):
		return Large_Scale_IPCMB(train_data,train_labels,thres)

	def small_scale_selector(self,train_data,train_labels,thres):
		return tian_IPCMB(train_data,train_labels,thres)[0]

	def get_threshold(self,max_loop):
		return self.get_threshold_bisection(max_loop)

class STMB(MBFeatureSelector):
	"Class that drives the algorithm for feature selection with STMB"
	def large_scale_selector(self,train_data,train_labels,thres):
		return Large_Scale_STMB(train_data,train_labels,self.indep_method, thres)

	def small_scale_selector(self,train_data,train_labels,thres):
		if self.indep_method == 'BayesFactor':
			selected_features = STMB_B_F(train_data, train_labels, self.bayesfactors)[0]
		else:
			selected_features = tian_STMB_new(train_data,train_labels,self.indep_method, thres)[0]
		return selected_features

	def get_threshold(self,max_loop):
		return self.get_threshold_bisection(max_loop)

#class STMB_BF(FeatureSelector):
#	"Class that drives the algorithm for feature selection with STMB"
#	def select_features(self):
#		train_data = self.train_set.data
#		train_labels = self.train_set.labels
#		selected_features = STMB_B_F(train_data, train_labels)[0]
#		return selected_features
    
class ASTMB(MBFeatureSelector):
	"Class that drives the algorithm for feature selection with aSTMB"
	def large_scale_selector(self,train_data,train_labels,thres):
		return self.small_scale_selector(train_data,train_labels,thres)

	def small_scale_selector(self,train_data,train_labels,thres):
		selected_features = Keyi_STMB_Adaptive_soft(train_data,train_labels,thres)[0]
		return selected_features

	def get_threshold(self,max_loop):
		return self.get_threshold_bisection(max_loop)

class S2TMB(FeatureSelector):
	"Class that contains the algorithm for feature selection with sSTMB"
	def select_features(self):
		train_data = self.train_set.data
		train_labels = self.train_set.labels
		test_data = self.test_set.data
		test_labels = self.test_set.labels
		Dtrain_data = self.discrete_train_set.data
		Dtrain_labels = self.discrete_train_set.labels
		return sSTMBplus(train_data,train_labels,test_data,test_labels,Dtrain_data,Dtrain_labels,self.problem_type)

class JMI(FeatureSelector):
	"Class that contains the algorithm for feature selection with JMI"
	def select_features(self,num_feats=None):
		if (num_feats==None):
			return self.get_optimal_num_feats()[1]
		train_data = self.train_set.data
		train_labels = self.train_set.labels
		return jmi(train_data,train_labels, num_feats, self.indep_method)

	def get_optimal_num_feats(self):
		"Use exhaustive search to find the optimal number of features in the reduced set (used for JMI)"
		# Initial optimal score will be computed with all features
		feats_to_accuracy = dict()
		num_feats = self.train_set.data.shape[1]
		if (num_feats>20):
			num_feats = 20
		optimal_num_feats = num_feats
		all_feats = self.select_features(num_feats=optimal_num_feats)
		optimal_feats = all_feats
		optimal_score = self.predictor.score(optimal_feats,feats_to_accuracy)
		for thres in range(1,num_feats):
			candidate_feats = all_feats[:thres]
			candidate_score = self.predictor.score(candidate_feats,feats_to_accuracy)
#			print(candidate_score)
			if (self.predictor.left_performs_better(candidate_score,candidate_feats,optimal_score,optimal_feats)):
				optimal_feats = candidate_feats
				optimal_score = candidate_score
		return optimal_feats.size,optimal_feats

class JMIplusSTMB(FeatureSelector):
	"Class that contains the algorithm for feature selection using JMI and STMB"
	def select_features(self):
		train_set = self.train_set
		test_set = self.test_set
		jmi_model = JMI(train_set,self.discrete_train_set,self.problem_type,test_set=test_set)
		jmi_feats = jmi_model.select_features()
		train_set = self.train_set.subset(jmi_feats)
		test_set = self.test_set.subset(jmi_feats)
		discrete_train_set = self.discrete_train_set.subset(jmi_feats)
		stmb_model = STMB(train_set,discrete_train_set,self.problem_type,test_set=test_set)
		stmb_thres = stmb_model.find_optimal_threshold(max_loop=3)[0]
		stmb_feats = stmb_model.select_features(thres=stmb_thres)
		return jmi_feats[stmb_feats]

class JMIplusASTMB(FeatureSelector):
	"Class that contains the algorithm for feature selection using JMI and STMB"
	def select_features(self):
		train_set = self.train_set
		test_set = self.test_set
		jmi_model = JMI(train_set,self.discrete_train_set,self.problem_type,test_set=test_set)
		jmi_feats = jmi_model.select_features()
		train_set = self.train_set.subset(jmi_feats)
		test_set = self.test_set.subset(jmi_feats)
		discrete_train_set = self.discrete_train_set.subset(jmi_feats)
		astmb_model = ASTMB(train_set,discrete_train_set,self.problem_type,test_set=test_set)
		astmb_thres = astmb_model.find_optimal_threshold(max_loop=3)[0]
		astmb_feats = astmb_model.select_features(thres=astmb_thres)
		return jmi_feats[astmb_feats]

class JMIplusS2TMB(FeatureSelector):
	"Class that contains the algorithm for feature selection using JMI and S2TMB"
	def select_features(self):
		train_set = self.train_set
		test_set = self.test_set
		jmi_model = JMI(train_set,self.discrete_train_set,self.problem_type,test_set=test_set)
		jmi_feats = jmi_model.select_features()
		train_set = self.train_set.subset(jmi_feats)
		test_set = self.test_set.subset(jmi_feats)
		discrete_train_set = self.discrete_train_set.subset(jmi_feats)
		s2tmb_model = S2TMB(train_set,discrete_train_set,self.problem_type,test_set=test_set)
		s2tmb_feats = s2tmb_model.select_features()
		return jmi_feats[s2tmb_feats]

    
