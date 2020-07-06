# with input in the DataFrame format
# update metadata of the dataset

import os, sys
import typing
import scipy.io
import numpy as np
from sklearn import preprocessing
from common_primitives import utils
from d3m import container
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.metadata import params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from typing import Optional, Sequence
import rpi_d3m_primitives
from rpi_d3m_primitives.featSelect.Feature_Selector_model import JMI
from rpi_d3m_primitives.featSelect.RelationSet import RelationSet
from sklearn.impute import SimpleImputer


Inputs = container.DataFrame
Outputs = container.DataFrame

__all__ = ('JMIplus_auto',)

class Params(params.Params):
    n_bins_: Optional[int]
    method_: Optional[str]
    strategy_: Optional[str]
    index_: Optional[Sequence[int]]
    problem_type_: Optional[str]


class Hyperparams(hyperparams.Hyperparams):
    nbins = hyperparams.UniformInt(
            lower=2,
            upper=21,
            default=10,
            description = 'The number of bins for discretization.',
            semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )
    method = hyperparams.Enumeration[str](
            values=['counting', 'pseudoBayesian','fullBayesian'],
            default='counting',
            description='The method for mutual information estimation.',
            semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )
    strategy = hyperparams.Enumeration[str](
            values=['uniform', 'quantile'],
            default='uniform',
            description='The method for KBins Discretizer.',
            semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )
    problem_type = hyperparams.Enumeration[str](
            values=['classification', 'regression'],
            default='classification',
            description='The task types',
            semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )


class JMIplus_auto(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that performs supervised feature selection to reduce input feature dimension. Input to this primitive should be a matrix of tabular numerical/categorical data, consisting of columns of features, and an array of labels. Output will be a reduced data matrix with metadata updated.
    """
    
    metadata = metadata_base.PrimitiveMetadata({
        'id': '4d46e77b-9c29-4631-b382-02333e928f10',
        'version': rpi_d3m_primitives.__coreversion__,
        'name': 'JMIplus_auto feature selector',
        'keywords': ['Joint Mutual Information','Feature Selection'],
        'description': 'This algorithm is selecting the most relevant features based on the joint mutual inforamtion between features and the target.',
        'source': {
            'name': rpi_d3m_primitives.__author__,
            'contact': 'mailto:yinn2@rpi.edu',
            'uris': [
                'https://github.com/zijun-rpi/d3m-primitives/blob/master/JMIplus_auto.py',
                'https://github.com/zijun-rpi/d3m-primitives.git'
                ]
        },
        'installation':[
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'rpi_d3m_primitives',
	            'version': rpi_d3m_primitives.__version__
            }
        ],
        'python_path': 'd3m.primitives.feature_selection.joint_mutual_information.AutoRPI',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.MINIMUM_REDUNDANCY_FEATURE_SELECTION
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, base.DockerContainer]] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        #Parameters
        self._index = None
        self._fitted = False
        #Hyperparameters & Parameters
        self._nbins = self.hyperparams['nbins']
        self._strategy = self.hyperparams['strategy']
        self._method = self.hyperparams['method']
        self._problem_type = self.hyperparams['problem_type']
        #other parameters
        self._training_inputs = None
        self._training_outputs = None
        self._cate_flag = None
        self._LEoutput = preprocessing.LabelEncoder()# label encoder
        self._Imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')#self.hyperparams['Imputer_Strategy']) # imputer
        self._Kbins = preprocessing.KBinsDiscretizer(n_bins=self._nbins, encode='ordinal', strategy=self._strategy) # KbinsDiscretizer
        
        
    ## TO DO:
    ## select columns via semantic types
    ## remove preprocessing
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        
        # set problem type
        metadata = outputs.metadata
        column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, 0))
        semantic_types = column_metadata.get('semantic_types', [])
        if self._problem_type == 'classification':
            m,n = outputs.shape
            outputs_copy = outputs.copy()
            if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' not in semantic_types:
                for i in range(m):
                    outputs_copy.iloc[i,0] = str(outputs.iloc[i,0])
            self._LEoutput.fit(outputs_copy)
            self._training_outputs = self._LEoutput.transform(outputs_copy)
            
        elif self._problem_type == 'regression':
            self._training_outputs = outputs.values

            
        # convert categorical values to numerical values in training data
        metadata = inputs.metadata
        [m,n] = inputs.shape
        self._training_inputs = np.zeros((m,n))
        self._cate_flag = np.zeros((n,))
        for column_index in metadata.get_elements((metadata_base.ALL_ELEMENTS,)):
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = column_metadata.get('semantic_types', [])
            if 'https://metadata.datadrivendiscovery.org/types/Attribute' in semantic_types and len(semantic_types) == 1:
                continue
            if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
                LE = preprocessing.LabelEncoder()
                LE = LE.fit(inputs.iloc[:,column_index])
                self._training_inputs[:,column_index] = LE.transform(inputs.iloc[:,column_index])
                self._cate_flag[column_index] = 1
            elif 'http://schema.org/Text' in semantic_types:
                pass #replaces with zeros
            else:
                temp = list(inputs.iloc[:, column_index].values)
                for i in np.arange(len(temp)):
                    if bool(temp[i]):
                        self._training_inputs[i,column_index] = float(temp[i])
                    else:
                        ## TO DO: float nan
                        self._training_inputs[i,column_index] = float('nan')
                if not np.count_nonzero(np.isnan(self._training_inputs[:, column_index])) == 0:  # if there is missing values
                    if np.count_nonzero(np.isnan(self._training_inputs[:,column_index])) == m:   # all missing
                        self._training_inputs[:,column_index] = np.zeros(m,) # replace with all zeros
                        
        self._fitted = False


    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs.any() == None or self._training_outputs.any() == None: 
            raise ValueError('Missing training data, or missing values exist.')

        ## impute missing values
        self._Imputer.fit(self._training_inputs)
        self._training_inputs = self._Imputer.transform(self._training_inputs)
        
#        [m,n] = self._training_inputs.shape
#        for column_index in range(n):
#            if len(np.unique(self._training_inputs[:,column_index])) == 1:
#                self._cate_flag[column_index] = 1
                
        ## discretize non-categorical values
        disc_training_inputs = self._training_inputs
        if not len(np.where(self._cate_flag == 0)[0]) == 0:
            self._Kbins.fit(self._training_inputs[:, np.where(self._cate_flag == 0)[0]]) #find non-categorical values
            temp = self._Kbins.transform(self._training_inputs[:, np.where(self._cate_flag == 0)[0]])
            disc_training_inputs[:, np.where(self._cate_flag == 0)[0]] = temp
        #start from zero
        
        Trainset = RelationSet(self._training_inputs, self._training_outputs.reshape(-1, 1))
        if self._problem_type == 'classification':
            discTrainset = RelationSet(disc_training_inputs, self._training_outputs.reshape(-1, 1))
        else: #regression
            self._Kbins.fit(self._training_outputs)
            disc_training_outputs = self._Kbins.transform(self._training_outputs)
            discTrainset = RelationSet(disc_training_inputs, disc_training_outputs)
#        discTrainset = RelationSet(disc_training_inputs, self._training_outputs.reshape(-1, 1))
        
        validSet, smallTrainSet = Trainset.split(self._training_inputs.shape[0] // 4)
        smallDiscTrainSet = discTrainset.split(self._training_inputs.shape[0] // 4)[1]
        model = JMI(smallTrainSet, smallDiscTrainSet,
                     self._problem_type, self._method, test_set=validSet)
        index = model.select_features()
        self._index = []
        [m, ] = index.shape
        for ii in np.arange(m):
            self._index.append(index[ii].item())
        self._fitted = True

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # inputs: m x n numpy array
        if self._fitted:
            output = inputs.iloc[:, self._index]
            output.metadata = utils.select_columns_metadata(inputs.metadata, columns=self._index)
            return CallResult(output)
        else:
            raise ValueError('Model should be fitted first.')


    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return Params(n_bins_ = self._nbins,
                      method_ = self._method,
                      strategy_ = self._strategy,
                      index_ = self._index,
                      problem_type_ = self._problem_type
        )


    def set_params(self, *, params: Params) -> None:
        self._nbins = params['n_bins_']
        self._method = params['method_']
        self._strategy = params['strategy_']
        self._index = params['index_']
        self._fitted = True
        self._problem_type = params['problem_type_']

