# AI Downloader

An easy-to-use, unified tool for downloading and managing all of your AI datasets and models.

## Datasets 

### Supported Catalogs & File Types
      
| Source | Info |
|----------|-----------|
| TensorFlow Datasets | [https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets) |
| Torchvision | [https://pytorch.org/vision/stable/datasets.html](https://pytorch.org/vision/stable/datasets.html) |
| Hugging Face | [https://huggingface.co/docs/datasets/index](https://huggingface.co/docs/datasets/index) |
| Generic Web URL | Publicly downloadable files: `.zip`, `.gz`, `.bz2`, `.txt`, `.csv`, `.png`, `.jpg`, etc. |

### Usage

Dataset catalog example:
```
from ai_downloader.datasets import DataDownloader

downloader = DataDownloader('tf_flowers', dataset_dir='/home/user/datasets', catalog='tensorflow_datasets')
downloader.download(split='train')
```

URL example:
```
from ai_downloader.datasets import DataDownloader

downloader = DataDownloader('my_dataset', dataset_dir='/home/user/datasets', url='http://<domain>/<filename>.zip')
downloader.download()
```

## Models

Coming soon.

## Build and Install

To install the downloader, follow [TLT's setup instructions](/README.md#build-and-install). The downloader is currently 
packaged alongside TLT and uses its requirements.txt files, but the tools can be separated at some future time. The 
downloader's dependencies are tracked in [requirements.txt](requirements.txt).

## Testing
With an activated environment that has the dependencies for the downloader and `pytest` in it, run this command from
the root repository directory:

```
py.test -s ai_downloader/tests
```
