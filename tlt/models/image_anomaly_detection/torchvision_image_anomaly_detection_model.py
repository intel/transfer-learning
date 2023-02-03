#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

from pydoc import locate
from numbers import Number
from tqdm import tqdm

import torch
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np

from tlt.utils.file_utils import verify_directory
from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel
from tlt.datasets.image_anomaly_detection.pytorch_custom_image_anomaly_detection_dataset \
    import PyTorchCustomImageAnomalyDetectionDataset


class TorchvisionImageAnomalyDetectionModel(TorchvisionImageClassificationModel):
    """
    Class to represent a Torchvision pretrained model for anomaly detection
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        TorchvisionImageClassificationModel.__init__(self, model_name, **kwargs)

    def _check_train_inputs(self, output_dir, dataset, dataset_type, pooling, kernel_size, pca_threshold):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if pooling not in ['avg', 'max']:
            raise TypeError("The specified pooling is not supported. It must be one of: {}".format(['avg', 'max']))

        if not isinstance(kernel_size, int):
            raise TypeError("The kernel_size  must be a {} but found a {}".format(int, type(kernel_size)))

        if not isinstance(pca_threshold, float) or pca_threshold <= 0.0 or pca_threshold >= 1.0:
            raise TypeError("The pca_threshold must be a float between 0 and 1  but found {}".format(pca_threshold))

    def extract_features(self, data, layer_name, pooling):
        """
        Extracts features of the layers specified using a layer name
        Args:
            data (torch.Tensor): Images to extract features from
            layer_name (string): The layer name whose output is desired for the extracted features
            pooling (list[string, int]): Pooling to be applied on the extracted layer ('avg' or 'max'), default is 'avg'

        Returns:
            outputs: Extracted features after applying pooling

        Raises:
            ValueError if the parameters are not within the expected values
        """
        pretrained_model_class = locate('torchvision.models.{}'.format(self.model_name))
        self._model = pretrained_model_class(pretrained=True)

        layer_names = [name for name, module in self._model.named_children()]
        if layer_name not in layer_names:
            raise TypeError("Invalid layer_name for the model. Choose from {}".format(layer_names))
        else:
            self._layer_name = layer_name

        self._model.eval()
        return_nodes = {layer: layer for layer in [layer_name]}
        partial_model = create_feature_extractor(self._model, return_nodes=return_nodes)
        features = partial_model(data)[layer_name]
        pooling_list = ['avg', 'max']
        if pooling[0] in pooling_list and pooling[0] == 'avg':
            pool_out = torch.nn.functional.avg_pool2d(features, pooling[1])
        elif pooling[0] in pooling_list and pooling[0] == 'max':
            pool_out = torch.nn.functional.max_pool2d(features, pooling[1])
        else:
            raise ValueError("The specified pooling is not supported")
        outputs = pool_out.contiguous().view(pool_out.size(0), -1)

        return outputs

    def train(self, dataset: PyTorchCustomImageAnomalyDetectionDataset, output_dir, do_eval=True, seed=None,
              layer_name='layer3', pooling='avg', kernel_size=2, pca_threshold=0.99):
        """
            Trains the model using the specified image anomaly detection dataset.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch
                seed (int): Optionally set a seed for reproducibility
                layer_name (str): The layer name whose output is desired for the extracted features
                pooling (str): Pooling to be applied on the extracted layer ('avg' or 'max'), default is 'avg'
                kernel_size (int): Kernel size in the pooling layer, default is 2
                pca_threshold (float): Threshold to apply to PCA model, default is 0.99

            Returns:
                Fitted principal components
        """
        self._check_train_inputs(output_dir, dataset, PyTorchCustomImageAnomalyDetectionDataset, pooling,
                                 kernel_size, pca_threshold)

        self._pooling = pooling
        self._kernel_size = kernel_size
        self._pca_threshold = pca_threshold

        images, labels = dataset.get_batch()
        outputs_inner = self.extract_features(images.to(self._device), layer_name, pooling=[pooling, kernel_size])
        data_mats_orig = torch.empty((outputs_inner.shape[1], len(dataset.train_subset))).to(self._device)

        # Feature extraction
        with torch.no_grad():
            data_idx = 0
            num_ims = 0
            for images, labels in tqdm(dataset._train_loader):
                images, labels = images.to(self._device), labels.to(self._device)
                num_samples = len(labels)
                outputs = self.extract_features(images, layer_name, pooling=[pooling, kernel_size])
                oi = torch.squeeze(outputs)
                data_mats_orig[:, data_idx:data_idx + num_samples] = oi.transpose(1, 0)
                num_ims += 1
                data_idx += num_samples

        # PCA
        data_mats_orig = data_mats_orig.numpy()
        self._pca_mats = PCA(pca_threshold)
        self._pca_mats.fit(data_mats_orig.T)

        if do_eval and dataset.validation_loader is not None:
            self.evaluate(dataset)

        return self._pca_mats

    def evaluate(self, dataset: PyTorchCustomImageAnomalyDetectionDataset, use_test_set=False):
        """
        Evaluate the accuracy of the model on a dataset.

        If there is a validation set, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, all of the good samples in the dataset will be
        used for evaluation.
        """
        if use_test_set:
            if dataset.test_subset:
                eval_loader = dataset.test_loader
                data_length = len(dataset.test_subset)
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset is not None:
            eval_loader = dataset.validation_loader
            data_length = len(dataset.validation_subset)
        else:
            eval_loader = dataset.data_loader
            data_length = len(dataset.dataset)

        if self._model is None:
            # The model hasn't been trained yet, use the original ImageNet trained model
            raise ValueError("The model has not been trained yet, so it can't be evaluated for anomaly detection")

        with torch.no_grad():
            gt = torch.zeros(data_length)
            scores = np.empty(data_length)
            count = 0
            for k, (images, labels) in enumerate(tqdm(eval_loader)):
                images = images.to(memory_format=torch.channels_last)
                num_im = images.shape[0]
                outputs = self.extract_features(images, self._layer_name, pooling=[self._pooling, self._kernel_size])
                feature_shapes = outputs.shape
                oi = outputs
                oi_or = oi
                oi_j = self._pca_mats.transform(oi)
                oi_reconstructed = self._pca_mats.inverse_transform(oi_j)
                fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
                fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
                scores[count: count + num_im] = -fre_score
                gt[count:count + num_im] = labels
                count += num_im

            gt = gt.numpy()

        fpr_binary, tpr_binary, _ = metrics.roc_curve(gt, scores)
        auc_roc_binary = metrics.auc(fpr_binary, tpr_binary)
        print(f'AUROC computed on {data_length} test images: {auc_roc_binary*100}')

        return auc_roc_binary

    def predict(self, input_samples, return_type='scores', threshold=None):
        """
        Perform inference and predict the class of the input_samples.

        Args:
            input_samples (tensor): Input tensor with one or more samples to perform inference on
            return_type (str): Using 'scores' will return the raw output of the PCA model and using 'class' will return
                               the highest scoring class based on a user-provided threshold (default: 'scores')
            threshold (numerical): Optional; When using return_type "class" this is the threshold for determining
                                   whether a score counts as an anomaly or not

        Returns:
            List of predictions ('good' or 'bad') or raw score vector
        """
        if return_type == 'class' and not isinstance(threshold, Number):
            raise ValueError("For class prediction, please give a numeric threshold.")

        with torch.no_grad():
            scores = np.empty(len(input_samples))
            count = 0
            input_samples = input_samples.to(memory_format=torch.channels_last)
            num_im = input_samples.shape[0]
            outputs = self.extract_features(input_samples, self._layer_name, pooling=[self._pooling, self._kernel_size])
            feature_shapes = outputs.shape
            oi = outputs
            oi_or = oi
            oi_j = self._pca_mats.transform(oi)
            oi_reconstructed = self._pca_mats.inverse_transform(oi_j)
            fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
            fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
            scores[count: count + num_im] = -fre_score
            count += num_im

        if return_type == 'scores':
            return scores
        else:
            return ['bad' if s < threshold else 'good' for s in scores]
