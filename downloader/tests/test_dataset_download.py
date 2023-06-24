import os
import pytest
import shutil
import tempfile

try:
    from datasets.arrow_dataset import Dataset as HF_Dataset
except ModuleNotFoundError:
    print("WARNING: datasets may not be installed")

try:
    from torch.utils.data import Dataset as TV_Dataset
except ModuleNotFoundError:
    print("WARNING: torch may not be installed")

try:
    from tensorflow.data import Dataset as TF_Dataset
except ModuleNotFoundError:
    print("WARNING: tensorflow may not be installed")

from downloader import datasets
from downloader.types import DatasetType


@pytest.mark.parametrize('dataset_name,catalog,url',
                         [['foo', 'tfds', 'https:...'],
                          ['bar', 'bar', None],
                          ['baz', None, None]])
def test_bad_download(dataset_name, catalog, url):
    """
    Tests downloader throws ValueError for bad inputs
    """
    with pytest.raises(ValueError):
        datasets.DataDownloader(dataset_name, dataset_dir='/tmp/data', catalog=catalog, url=url)


class TestDatasetDownload:
    """
    Tests the dataset downloader with a temp download directory that is initialized and cleaned up
    """
    URLS = {'sms_spam_collection':
            'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',
            'flowers':
            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
            'imagenet_labels':
            'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt',
            'peacock':
            'https://c8.staticflickr.com/8/7095/7210797228_c7fe51c3cb_z.jpg',
            'pennfudan':
            'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'}

    @classmethod
    def setup_class(cls):
        cls._dataset_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._dataset_dir):
            print("Deleting test directory:", cls._dataset_dir)
            shutil.rmtree(cls._dataset_dir)

    @pytest.mark.integration
    @pytest.mark.parametrize('dataset_name,catalog,split,kwargs,size',
                             [['tf_flowers', 'tfds', 'train', {}, 3670],
                              ['CIFAR10', 'torchvision', 'train', {}, 50000],
                              ['CIFAR10', 'torchvision', 'val', {}, 10000],
                              ['imdb', 'huggingface', 'train', {}, 25000],
                              ['glue', 'huggingface', 'test', {'subset': 'sst2'}, 1821]])
    def test_catalog_download(self, dataset_name, catalog, split, kwargs, size):
        """
        Tests downloader for different dataset catalog types and splits
        """
        downloader = datasets.DataDownloader(dataset_name, dataset_dir=self._dataset_dir, catalog=catalog, **kwargs)
        data = downloader.download(split=split)

        # Check the type of the downloader and returned object
        if catalog == 'tfds':
            data = data[0]  # TFDS returns a list with the dataset in it
            assert downloader._type == DatasetType.TENSORFLOW_DATASETS
            assert isinstance(data, TF_Dataset)
        elif catalog == 'torchvision':
            assert downloader._type == DatasetType.TORCHVISION
            assert isinstance(data, TV_Dataset)
        elif catalog == 'huggingface':
            assert downloader._type == DatasetType.HUGGING_FACE
            assert isinstance(data, HF_Dataset)

        # Verify the split size
        assert len(data) == size

        # Check that the directory is not empty
        assert os.listdir(self._dataset_dir) is not None

    @pytest.mark.parametrize('dataset_name,url,num_contents',
                             [['sms_spam_collection', URLS['sms_spam_collection'], 2],
                              ['flowers', URLS['flowers'], 1],
                              ['imagenet_labels', URLS['imagenet_labels'], 1],
                              ['peacock', URLS['peacock'], 1],
                              ['pennfudan', URLS['pennfudan'], 1]])
    def test_generic_download(self, dataset_name, url, num_contents):
        """
        Tests downloader for different web URLs and file types
        """
        downloader = datasets.DataDownloader(dataset_name, dataset_dir=self._dataset_dir, url=url)
        data_path = downloader.download()

        assert downloader._type == DatasetType.GENERIC

        # Test that the returned object is the expected type and length
        if num_contents == 1:
            assert isinstance(data_path, str)
            assert os.path.exists(data_path)
        else:
            assert isinstance(data_path, list)
            for path in data_path:
                assert os.path.exists(path)
