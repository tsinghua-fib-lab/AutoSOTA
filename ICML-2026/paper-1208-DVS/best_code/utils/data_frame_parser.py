### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
from logging import getLogger

import numpy
from rdkit import Chem
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.getcwd())
from .smile_to_graph import GGNNPreprocessor, MolFeatureExtractionError
from .numpytupledataset import NumpyTupleDataset
import traceback


# Code adapted from chainer_chemistry\dataset\parsers\data_frame_parser.py
class DataFrameParser(object):
    def __init__(self, preprocessor,
                 labels=None,
                 smiles_col='smiles',
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(DataFrameParser, self).__init__()
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels
        self.smiles_col = smiles_col
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)
        self.preprocessor = preprocessor

    def parse(self, df, return_smiles=False, target_index=None,
              return_is_successful=False):
        logger = self.logger
        pp = self.preprocessor
        smiles_list = []
        is_successful_list = []

        # counter = 0
        if isinstance(pp, GGNNPreprocessor):
            if target_index is not None:
                df = df.iloc[target_index]

            features = None
            smiles_index = df.columns.get_loc(self.smiles_col)
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles = row[smiles_index]
                # TODO(Nakago): Check.
                # currently it assumes list
                labels = [row[i] for i in labels_index]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        fail_count += 1
                        if return_is_successful:
                            is_successful_list.append(False)
                        continue
                    # Note that smiles expression is not unique.
                    # we obtain canonical smiles
                    canonical_smiles, mol = pp.prepare_smiles_and_mol(mol)
                    input_features = pp.get_input_features(mol)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels = self.postprocess_label(labels)

                    if return_smiles:
                        smiles_list.append(canonical_smiles)
                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'.format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                # Initialize features: list of list
                if features is None:
                    if isinstance(input_features, tuple):
                        num_features = len(input_features)
                    else:
                        num_features = 1
                    if self.labels is not None:
                        num_features += 1
                    features = [[] for _ in range(num_features)]

                if isinstance(input_features, tuple):
                    for i in range(len(input_features)):
                        features[i].append(input_features[i])
                else:
                    features[0].append(input_features)
                if self.labels is not None:
                    features[len(features) - 1].append(labels)
                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)
            ret = []

            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = numpy.empty(len(feature), dtype=numpy.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        smileses = numpy.array(smiles_list) if return_smiles else None
        if return_is_successful:
            is_successful = numpy.array(is_successful_list)
        else:
            is_successful = None

        if isinstance(result, (tuple, list)):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            dataset = NumpyTupleDataset(result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            dataset = NumpyTupleDataset([result])

        return {"dataset": dataset,
                "smiles": smileses,
                "is_successful": is_successful}

    def extract_total_num(self, df):
        return len(df)
