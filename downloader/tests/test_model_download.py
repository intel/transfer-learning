import os
import pytest
import shutil
import tempfile

try:
    from torch.nn import Module
except ModuleNotFoundError:
    print("WARNING: Unable to import torch. Torch may not be installed")

try:
    from tensorflow_hub.keras_layer import KerasLayer
except ModuleNotFoundError:
    print("WARNING: Unable to import KerasLayer. Tensorflow Hub may not be installed")

try:
    from tensorflow.keras import Model
except ModuleNotFoundError:
    print("WARNING: Unable to import Keras Model. Tensorflow may not be installed")

from downloader import models
from downloader.types import ModelType


@pytest.mark.parametrize('hub',
                         [['foo'],
                          ['bar'],
                          ['baz']])
def test_bad_hub(hub):
    """
    Tests downloader throws ValueError for bad inputs
    """
    model_name = 'model'
    with pytest.raises(ValueError):
        models.ModelDownloader(model_name, hub)


class TestModelDownload:
    """
    Tests the model downloader with a temp download directory that is initialized and cleaned up
    """
    @classmethod
    def setup_class(cls):
        cls._model_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._model_dir):
            print("Deleting test directory:", cls._model_dir)
            shutil.rmtree(cls._model_dir)

    # Has previously been skipped due to HTTP Error 403: rate limit exceeded')
    @pytest.mark.parametrize('model_name,hub,kwargs',
                             [['https://tfhub.dev/google/efficientnet/b0/feature-vector/1', 'tf_hub', {}],
                              ['https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', 'tfhub',
                               {'name': 'encoder', 'trainable': True}],
                              ['resnet34', 'torchvision', {}],
                              ['mobilenet_v2', 'torchvision', {}],
                              ['resnet18_ssl', 'pytorch_hub', {}],
                              ['resnet50_swsl', 'pytorch_hub', {}],
                              ['distilbert-base-uncased', 'huggingface', {}],
                              ['bert-base-cased', 'hugging_face', {}],
                              ['Xception', 'keras_applications', {}],
                              ['ResNet50', 'keras', {'weights': 'imagenet', 'include_top': False}],
                              ['google/bert_uncased_L-2_H-128_A-2', 'tf_bert_huggingface', {}],
                              ['bert-base-uncased', 'tf_bert_hugging_face', {}]])
    def test_hub_download(self, model_name, hub, kwargs):
        """
        Tests downloader for different model hubs
        """
        downloader = models.ModelDownloader(model_name, hub, model_dir=self._model_dir, **kwargs)
        model = downloader.download()

        # Check the type of the downloader and returned object
        if downloader._type == ModelType.TF_HUB:
            assert isinstance(model, KerasLayer)
        elif downloader._type == ModelType.TORCHVISION:
            assert isinstance(model, Module)
        elif downloader._type == ModelType.PYTORCH_HUB:
            assert isinstance(model, Module)
        elif downloader._type == ModelType.HUGGING_FACE:
            assert isinstance(model, Module)
        elif downloader._type == ModelType.KERAS_APPLICATIONS:
            assert isinstance(model, Model)
        elif downloader._type == ModelType.TF_BERT_HUGGINGFACE:
            assert isinstance(model, Model)
        else:
            assert False

        # Check that the directory is not empty
        assert os.listdir(self._model_dir) is not None
