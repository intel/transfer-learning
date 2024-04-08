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
# SPDX-License-Identifier: Apache-2.0
#

from numbers import Number
import pickle  # nosec B403
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch.nn as nn
import os
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

from downloader.models import ModelDownloader
from tlt.models.image_classification.pytorch_image_classification_model import PyTorchImageClassificationModel
from tlt.datasets.image_anomaly_detection.pytorch_custom_image_anomaly_detection_dataset \
    import PyTorchCustomImageAnomalyDetectionDataset
from tlt.utils.file_utils import verify_directory, validate_model_name
from tlt.utils.platform_util import PlatformUtil
from tlt.utils.types import UseCaseType

from tlt.models.image_anomaly_detection.simsiam import builder
from tlt.models.image_anomaly_detection import utils

from tlt.models.image_anomaly_detection.cutpaste.model import ProjectionNet
from tlt.models.image_anomaly_detection.cutpaste.cutpaste import CutPasteNormal, \
    CutPasteScar, CutPaste3Way, CutPasteUnion


try:
    habana_import_error = None
    import habana_frameworks.torch.core as htcore
    is_hpu_available = True
except Exception as e:
    is_hpu_available = False
    habana_import_error = str(e)


def extract_features(model, data, layer_name, pooling):
    """
    Extracts features of the layers specified using a layer name
    Args:
        model (PyTorchImageAnomalyDetectionModel/SimSiam/ProjectionNet): Model on which features
                                                                         are to be extracted
        data (torch.Tensor): Images to extract features from
        layer_name (string): The layer name whose output is desired for the extracted features
        pooling (list[string, int]): Pooling to be applied on the extracted layer ('avg' or 'max'), default is 'avg'

    Returns:
        outputs: Extracted features after applying pooling

    Raises:
        ValueError if the parameters are not within the expected values
    """
    features = model(data)[layer_name]
    pooling_list = ['avg', 'max']
    if pooling[0] in pooling_list and pooling[0] == 'avg':
        pool_out = torch.nn.functional.avg_pool2d(features, pooling[1])
    elif pooling[0] in pooling_list and pooling[0] == 'max':
        pool_out = torch.nn.functional.max_pool2d(features, pooling[1])
    else:
        raise ValueError("The specified pooling is not supported")
    outputs = pool_out.contiguous().view(pool_out.size(0), -1)

    return outputs


def get_feature_extraction_model(model, layer_name):
    """
    Get partial model split by a specified layer name
    Args:
        model (PyTorchImageAnomalyDetectionModel/SimSiam/ProjectionNet): Model on which features have to be extracted
        layer_name (string): The layer name whose output is desired for the extracted features
    Returns:
        outputs: Partial model

    Raises:
        ValueError if the parameters are not within the expected values
    """
    layer_names = [name for name, module in model.named_children()]
    if layer_name not in layer_names:
        raise TypeError("Invalid layer_name for the model. Choose from {}".format(layer_names))

    model.eval()
    return_nodes = {layer: layer for layer in [layer_name]}
    partial_model = create_feature_extractor(model, return_nodes=return_nodes)
    return partial_model


def pca(features, threshold=0.99):
    """
    Finds the principal components of the features specified.

        Args:
            features (torch.Tensor): Input features
            threshold (float): Threshold to apply to PCA model, default is 0.99

        Returns:
            PCA components responsible for the top (threshold)% of the features' variability
    """
    # Converting features to cpu just in case hpu is used, hpu tensors cannot use numpy()
    features = features.cpu().numpy()
    principal_components = PCA(threshold)
    pca_mats = principal_components.fit(features.T)
    return pca_mats


class PyTorchImageAnomalyDetectionModel(PyTorchImageClassificationModel):
    """
    Class to represent a PyTorch model for image classification
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):
        """
        Class constructor
        """
        PyTorchImageClassificationModel.__init__(self, model_name, model, optimizer, loss,
                                                 use_case=UseCaseType.IMAGE_ANOMALY_DETECTION, **kwargs)
        self.simsiam = False
        self.cutpaste = False
        self._layer_name = 'layer3'
        self._pooling = 'avg'
        self._kernel_size = 2
        self._pca_mats = None
        self._hub = 'torchvision'
        self._enable_auto_mixed_precision = False
        self._device = kwargs.get("device", "cpu")

        # Store the dataset type that this model type can use for Intel Neural Compressor
        self._inc_compatible_dataset = (PyTorchCustomImageAnomalyDetectionDataset)

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

    def load_checkpoint_weights(self, model_name, checkpoint_dir, filename, feature_extractor=None, hub=None):
        """
        Load checkpoints from the given checkpoint directory based on feature extractor
        """
        if hub is None:
            hub = self._hub

        downloader = ModelDownloader(model_name, hub=hub, model_dir=None)
        net = downloader.download()

        ckpt = torch.load(os.path.join(checkpoint_dir, filename), map_location=torch.device(self._device))
        state_dict = ckpt['state_dict']

        if feature_extractor == 'simsiam':
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('encoder.') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif feature_extractor == 'cutpaste':
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('model.'):
                    # remove prefix
                    state_dict[k[len("model."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        else:
            head_layers = [512] * self.head_layer + [128]
            num_classes = state_dict["out.weight"].shape[0]
            net = ProjectionNet(model_name=model_name, pretrained=False,
                                head_layers=head_layers, num_classes=num_classes)
            train_nodes, eval_nodes = get_graph_node_names(net)

        # load params
        net.load_state_dict(state_dict, strict=False)
        return net

    def load_pretrained_model(self):
        """
            Return PyTorchImageAnomalyDetectionModel object for
            feature extraction and finding PCA

            Returns:
                Model object
        """
        return self._model

    def train_simsiam(self, dataset, output_dir, epochs, feature_dim,
                      pred_dim, batch_size=64, initial_checkpoints=None,
                      generate_checkpoints=False, ipex_optimize=True, precision='float32',
                      hub=None):
        """
            Trains a SimSiam model using the specified dataset.

            Args:
                dataset (str): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                epochs (int): Number of epochs to train the model
                feature_dim (int): feature dimension
                pred_dim (int): hidden dimension of the predictor
                initial_checkpoints (str): Path to checkpoint weights to load.
                generate_checkpoints (bool): Whether to save/preserve the best weights during
                                             SimSiam or CutPaste training, default is False.
                ipex_optimize (bool): Use Intel Extension for PyTorch (IPEX). Defaults to True.
                precision (str): precision in which model to be trained, default is float32.

            Returns:
                Model object

        """
        self.LR = 0.171842137353148
        self.batch_size = batch_size
        self.batch_size_ss = 64
        self.epochs = epochs
        self.simsiam = True

        if hub is None:
            hub = self._hub

        dataset._dataset.transform = dataset._simsiam_transform
        dataloader = dataset._train_loader

        print("Creating SIMSIAM feature extractor with the backbone of '{}'".format(self.model_name))

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints, map_location='cpu')
            self._model.load_state_dict(checkpoint, strict=False)
        self._model = builder.SimSiam(self._model, feature_dim, pred_dim)
        self._model = self._model.to(self._device)
        init_lr = self.LR * self.batch_size / 256

        criterion = nn.CosineSimilarity(dim=1).to(self._device)

        optim_params = [{'params': self._model.encoder.parameters(), 'fix_lr': False},
                        {'params': self._model.predictor.parameters(), 'fix_lr': True}]
        optimizer = torch.optim.SGD(optim_params, init_lr, momentum=0.9, weight_decay=1e-4)
        num_images = len(dataset.train_subset)

        best_least_Loss = float('inf')
        is_best_ans = False
        file_name_least_loss = ""
        print("Fine-tuning Simsiam Model on ", epochs, "epochs using ", num_images, " training images")
        self._model.train()

        if self._device == "hpu" and ipex_optimize:
            # Gaudi is not compatible with IPEX
            print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
            ipex_optimize = False

        if ipex_optimize:
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(self._model, optimizer=optimizer,
                                             dtype=torch.bfloat16 if precision == 'bfloat16' else torch.float32)
        else:
            model = self._model

        valid_model_name = validate_model_name(self.model_name)
        checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
        verify_directory(checkpoint_dir)

        for epoch in range(0, self.epochs):
            utils.adjust_learning_rate(optimizer, init_lr, epoch, self.epochs)

            curr_loss = utils._fit_simsiam(dataloader, model, criterion, optimizer, epoch, precision, self._device)
            if generate_checkpoints:
                if (curr_loss < best_least_Loss):
                    best_least_Loss = curr_loss
                    is_best_ans = True
                    file_name_least_loss = 'simsiam_checkpoint_{:04d}.pth.tar'.format(epoch)

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best_ans, file_name_least_loss,
                    best_least_Loss, checkpoint_dir)
                is_best_ans = False
            else:
                if epoch == self.epochs - 1:
                    file_name_least_loss = 'simsiam_checkpoint_{:04d}.pth.tar'.format(epoch)
                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': self.model_name,
                        'state_dict': model.state_dict(),
                    }, is_best=True, filename=file_name_least_loss,
                        loss=curr_loss, checkpoint_dir=checkpoint_dir)

        print('No. Of Epochs=', self.epochs)
        print('Batch Size =', self.batch_size_ss)
        self._model = self.load_checkpoint_weights(self.model_name, checkpoint_dir,
                                                   file_name_least_loss, feature_extractor='simsiam',
                                                   hub=hub)
        return self._model

    def train_cutpaste(self, dataset, output_dir, optim, epochs, freeze_resnet,
                       head_layer, cutpaste_type, initial_checkpoints=None,
                       generate_checkpoints=False, ipex_optimize=True, precision='float32',
                       hub=None):
        """
            Trains a CutPaste model using the specified dataset.

            Args:
                dataset (str): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                optim (str): Choice of optimizer to use for training
                epochs (int): Number of epochs to train the model
                freeze_resnet (int): Epochs upto which we freeze ResNet layers
                                     and only train the new header with FC layers
                head_layer (int): number of layers in the projection head
                cutpaste-type (str): cutpaste variant to use
                initial_checkpoints (str): path for feature extractor model
                generate_checkpoints (bool): Whether to save/preserve the best weights during
                                             SimSiam or CutPaste training, default is False.
                ipex_optimize (bool): Use Intel Extension for PyTorch (IPEX). Defaults to True.
                precision (str): precision in which model to be trained, default is float32.

            Returns:
                Model object

        """
        self.variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar,
                            '3way': CutPaste3Way, 'union': CutPasteUnion}
        variant = self.variant_map[cutpaste_type]
        self.cutpaste = True

        dataset._dataset.transform = dataset._cutpaste_transform
        dataloader = dataset._train_loader

        valid_model_name = validate_model_name(self.model_name)
        checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
        verify_directory(checkpoint_dir)

        if hub is None:
            hub = self._hub

        if initial_checkpoints is None:
            print("=> creating CUT-PASTE feature extractor with the backbone of'{}'".format(self.model_name))

            # create Model:
            head_layers = [512] * head_layer + [128]
            num_classes = 2 if variant is not CutPaste3Way else 3
            self._model = ProjectionNet(model_name=self.model_name, pretrained=True,
                                        head_layers=head_layers, num_classes=num_classes)
            self._model = self._model.to(self._device)
            if freeze_resnet > 0:
                self._model.freeze_resnet()

            criterion = torch.nn.CrossEntropyLoss()
            weight_decay = 0.00003
            momentum = 0.9
            if optim == "sgd":
                optimizer = torch.optim.SGD(self._model.parameters(), lr=0.03, momentum=momentum,
                                            weight_decay=weight_decay)
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs)
            elif optim == "adam":
                optimizer = torch.optim.Adam(self._model.parameters(), lr=0.03, weight_decay=weight_decay)
                scheduler = None
            else:
                print(f"ERROR unkown optimizer: {optim}")

            num_images = len(dataset.train_subset)
            best_least_Loss = float('inf')
            is_best_ans = False
            file_name_least_loss = ""
            print("Fine-tuning CUT-PASTE Model on ", epochs, "epochs using ", num_images, " training images")
            self._model.train()

            if self._device == "hpu" and ipex_optimize:
                # Gaudi is not compatible with IPEX
                print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
                ipex_optimize = False

            if ipex_optimize:
                import intel_extension_for_pytorch as ipex
                model, optimizer = ipex.optimize(self._model, optimizer=optimizer,
                                                 dtype=torch.bfloat16 if precision == 'bfloat16' else torch.float32)
            else:
                model = self._model

            for step in range(epochs):
                epoch = int(step / 1)
                curr_loss = utils._fit_cutpaste(dataloader, model, criterion,
                                                optimizer, epoch, freeze_resnet, scheduler, precision, self._device)
                if generate_checkpoints:
                    if (curr_loss < best_least_Loss):
                        best_least_Loss = curr_loss
                        is_best_ans = True
                        file_name_least_loss = 'cutpaste_checkpoint_{:04d}.pth.tar'.format(step)

                    # Saves the Best Intermediate Checkpoints got till this step.
                    utils.save_checkpoint({
                        'epoch': step + 1,
                        'arch': self.model_name,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best=is_best_ans, filename=file_name_least_loss,
                        loss=best_least_Loss, checkpoint_dir=checkpoint_dir)
                    is_best_ans = False
                else:
                    if epoch == epochs - 1:
                        file_name_least_loss = 'cutpaste_checkpoint_{:04d}.pth.tar'.format(step)
                        utils.save_checkpoint({
                            'epoch': step + 1,
                            'arch': self.model_name,
                            'state_dict': model.state_dict(),
                        }, is_best=True, filename=file_name_least_loss,
                            loss=curr_loss, checkpoint_dir=checkpoint_dir)
            self._model = self.load_checkpoint_weights(self.model_name, checkpoint_dir,
                                                       file_name_least_loss, feature_extractor='cutpaste',
                                                       hub=hub)
        else:
            models = []
            for filename in os.listdir(initial_checkpoints):
                f = os.path.join(initial_checkpoints, filename)
                # checking if it is a file and correct self-supervised technique
                if os.path.isfile(f) and self.cutpaste:
                    models.append(f)
            model_path = os.path.basename(max(models, key=os.path.getctime))
            self._model = self.load_checkpoint_weights(self.model_name, checkpoint_dir,
                                                       model_path, feature_extractor='cutpaste',
                                                       hub=hub)
        return self._model

    def train(self, dataset: PyTorchCustomImageAnomalyDetectionDataset, output_dir, epochs=1,
              batch_size=64, feature_dim=1000, pred_dim=250,
              generate_checkpoints=False, initial_checkpoints=None, seed=None, pooling='avg',
              kernel_size=2, pca_threshold=0.99, simsiam=False, cutpaste=False, cutpaste_type='normal',
              freeze_resnet=20, head_layer=2, optim='sgd', layer_name='layer3', ipex_optimize=True,
              enable_auto_mixed_precision=None, device=None):
        """
            Trains the model using the specified image anomaly detection dataset.

            Args:
                dataset (PyTorchCustomImageAnomalyDetectionDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                batch_size (int): batch size for every forward operation, default is 64
                layer_name (str): The layer name whose output is desired for the extracted features
                feature_dim (int): Feature dimension, default is 1000
                pred_dim (int): Hidden dimension of the predictor, default is 250
                epochs (int): Number of epochs to train the model
                generate_checkpoints (bool): Whether to save/preserve the best weights during
                                             SimSiam or CutPaste training, default is False.
                initial_checkpoints (str): Path to checkpoint weights to load
                seed (int): Optional, set a seed for reproducibility
                pooling (str): Pooling to be applied on the extracted layer ('avg' or 'max'), default is 'avg'
                kernel_size (int): Kernel size in the pooling layer, default is 2
                pca_threshold (float): Threshold to apply to PCA model, default is 0.99
                simsiam (bool): Boolean option to enable/disable simsiam training, default is False
                cutpaste (bool): Boolean option to enable/disable cutpaste training, default is False
                cutpaste_type (str): cutpaste variant to use, default is normal
                freeze_resnet (int): Epochs up to which we freeze ResNet layers and only train
                                     the new header with FC layers, default is 20
                head_layer (int): number of layers in the projection head, default is 2
                optim (str): Choice of optimizer to use for training, default is sgd
                ipex_optimize (bool): Use Intel Extension for PyTorch (IPEX). Defaults to True.
                enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms. Defaults to
                    None.
                device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")

            Returns:
                Fitted principal components and PyTorch feature extraction model
        """
        self._check_train_inputs(output_dir, dataset, PyTorchCustomImageAnomalyDetectionDataset, pooling,
                                 kernel_size, pca_threshold)

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        self._pooling = pooling
        self._kernel_size = kernel_size
        self._pca_threshold = pca_threshold
        self._layer_name = layer_name
        self.batch_size = batch_size
        self.simsiam = simsiam
        self.cutpaste = cutpaste

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"

        if self._device == "hpu" and ipex_optimize:
            # Gaudi is not compatible with IPEX
            print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
            ipex_optimize = False

        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))

        self._enable_auto_mixed_precision = enable_auto_mixed_precision
        precision = 'float32' if not enable_auto_mixed_precision else 'bfloat16'

        if self.simsiam:
            model = self.train_simsiam(dataset, output_dir, epochs, feature_dim,
                                       pred_dim, batch_size, initial_checkpoints,
                                       generate_checkpoints=False, ipex_optimize=ipex_optimize,
                                       precision=precision, hub=self._hub)
        elif self.cutpaste:
            model = self.train_cutpaste(dataset, output_dir, optim, epochs, freeze_resnet,
                                        head_layer, cutpaste_type, initial_checkpoints,
                                        generate_checkpoints=False, ipex_optimize=ipex_optimize,
                                        precision=precision, hub=self._hub)
        else:
            model = self.load_pretrained_model()
            print("Loading '{}' model".format(self.model_name))

        model = model.to(self._device)
        model = get_feature_extraction_model(model, layer_name)
        dataset._dataset.transform = dataset._train_transform
        images, labels = dataset.get_batch()
        outputs_inner = extract_features(model, images.to(self._device), layer_name,
                                         pooling=[pooling, kernel_size])
        data_mats_orig = torch.empty((outputs_inner.shape[1], len(dataset.train_subset))).to(self._device)

        # Feature extraction
        with torch.no_grad():
            data_idx = 0
            num_ims = 0
            for images, labels in tqdm(dataset._train_loader):
                images, labels = images.to(self._device), labels.to(self._device)
                num_samples = len(labels)
                outputs = extract_features(model, images, layer_name, pooling=[pooling, kernel_size])
                oi = torch.squeeze(outputs)
                data_mats_orig[:, data_idx:data_idx + num_samples] = oi.transpose(1, 0)
                num_ims += 1
                data_idx += num_samples

        # PCA
        self._pca_mats = pca(data_mats_orig, pca_threshold)

        return self._pca_mats, model

    def evaluate(self, dataset: PyTorchCustomImageAnomalyDetectionDataset, pca_mats=None, use_test_set=False,
                 device=None):
        """
        Evaluate the accuracy of the model on a dataset.
        If there is a validation set, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, all of the good samples in the dataset will be
        used for evaluation.

        Args:
            dataset (PyTorchCustomImageAnomalyDetectionDataset): Dataset on which evaluation
                                                                 is performed
            pca_mats (PCA components): Components responsible for the top (threshold)% of the
                                       features' variability; if not provided, the model's last
                                       calculated PCA will be used
            use_test_set (bool): If set to True, evaluation is done on test set else on validation
                                 set if available
            device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")


        Returns:
            threshold : Computed threshold for prediction
            auc_roc_binary : AUROC score
        """
        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"

        if pca_mats is None:
            if self._pca_mats is None:
                raise ValueError("Either pass in the pca_mats to use or use the train() method to generate them.")
            else:
                pca_mats = self._pca_mats

        dataset._dataset.transform = dataset._validation_transform
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

        print("Evaluating on {} test images".format(data_length))

        model = get_feature_extraction_model(self._model, self._layer_name)
        with torch.no_grad():
            gt = torch.zeros(data_length)
            scores = np.empty(data_length)
            count = 0
            for k, (images, labels) in enumerate(tqdm(eval_loader)):
                images = images.to(device=self._device, memory_format=torch.channels_last)
                num_im = images.shape[0]
                outputs = extract_features(model, images, self._layer_name,
                                           pooling=[self._pooling, self._kernel_size])
                feature_shapes = outputs.shape
                oi = outputs
                oi_or = oi.cpu()
                oi_j = pca_mats.transform(oi.cpu())
                oi_reconstructed = pca_mats.inverse_transform(oi_j)
                fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
                fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
                scores[count: count + num_im] = -fre_score
                gt[count:count + num_im] = labels
                count += num_im
                if self._device == "hpu" and is_hpu_available:
                    htcore.mark_step()

            gt = gt.numpy()

        fpr_binary, tpr_binary, thres = metrics.roc_curve(gt, scores)
        threshold = utils.find_threshold(fpr_binary, tpr_binary, thres)
        auc_roc_binary = metrics.auc(fpr_binary, tpr_binary)
        accuracy_score = metrics.accuracy_score(gt, [1 if i >= threshold else 0 for i in scores])
        print(f'AUROC: {auc_roc_binary * 100}')
        print(f'Accuracy: {accuracy_score * 100}')

        return threshold, auc_roc_binary

    def predict(self, input_samples, pca_mats=None, return_type='scores', threshold=None, device=None):
        """
        Perform inference and predict the class of the input_samples.

        Args:
            input_samples (tensor): Input tensor with one or more samples to perform inference on
            pca_mats (PCA components): Components responsible for the top (threshold)% of the
                                       features' variability; if not provided, the model's last
                                       calculated PCA will be used
            return_type (str): Using 'scores' will return the raw output of the PCA model and using 'class' will return
                               the highest scoring class based on a user-provided threshold (default: 'scores')
            threshold (numerical): Optional; When using return_type "class" this is the threshold for determining
                                   whether a score counts as an anomaly or not
            device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")

        Returns:
            List of predictions ('good' or 'bad') or raw score vector
        """
        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"

        if pca_mats is None:
            if self._pca_mats is None:
                raise ValueError("Either pass in the pca_mats to use or use the train() method to generate them.")
            else:
                pca_mats = self._pca_mats

        if return_type == 'class' and not isinstance(threshold, Number):
            raise ValueError("For class prediction, please give a numeric threshold.")

        model = get_feature_extraction_model(self._model, self._layer_name)
        with torch.no_grad():
            scores = np.empty(len(input_samples))
            count = 0
            input_samples = input_samples.to(device=self._device, memory_format=torch.channels_last)
            num_im = input_samples.shape[0]
            outputs = extract_features(model, input_samples, self._layer_name,
                                       pooling=[self._pooling, self._kernel_size])
            feature_shapes = outputs.shape
            oi = outputs
            oi_or = oi.cpu()
            oi_j = pca_mats.transform(oi.cpu())
            oi_reconstructed = pca_mats.inverse_transform(oi_j)
            fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
            fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
            scores[count: count + num_im] = -fre_score
            count += num_im
            if self._device == "hpu" and is_hpu_available:
                htcore.mark_step()

        if return_type == 'scores':
            return scores
        else:
            return ['good' if s >= threshold else 'bad' for s in scores]

    def export(self, output_dir):
        """
           Exports a trained model as a model.pt file along with the PCA components as pca_mats.pkl. The files
           will be written to the output directory in a directory with the model's name, and a unique numbered
           directory. The directory number will increment each time the model is exported.

           Args:
               output_dir (str): A writeable output directory.

           Returns:
               The path to the numbered saved model directory

           Raises:
               TypeError: if the output_dir is not a string
               FileExistsError: the specified output directory already exists as a file
               ValueError: if the model has not been loaded or trained yet
        """
        # Call the base class to write the model file
        saved_model_dir = PyTorchImageClassificationModel.export(self, output_dir)

        if self._pca_mats:
            pca_file_name = os.path.join(saved_model_dir, 'pca_mats.pkl')
            with open(pca_file_name, 'wb') as pca_file:
                pickle.dump(self._pca_mats, pca_file)
            print('Saved principal components: {}'.format(pca_file_name))

        return saved_model_dir

    def load_from_directory(self, model_dir: str):
        """
        Load a saved model and its PCA components from the model_dir path
        """
        # Call the base class to load the model file
        PyTorchImageClassificationModel.load_from_directory(self, model_dir)

        pca_file_name = os.path.join(model_dir, 'pca_mats.pkl')
        if os.path.isfile(pca_file_name):
            with open(pca_file_name, 'rb') as pca_file:
                self._pca_mats = pickle.load(pca_file)  # nosec B301
