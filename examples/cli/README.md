# Intel® Transfer Learning Tool CLI Examples

The following example walks through a full workflow using the Intel Transfer Learning
Tool CLI to train a model, and then benchmark, quantize, and optimize the
trained model. It uses a TensorFlow image classification model, but the
same commands and concepts can be applied when working with other frameworks
and use cases.

Use `tlt --help` to see the list of CLI commands. More detailed information on
each command can be found using `tlt <command> --help` (like `tlt train --help`).

**List the available models**:
Use the `tlt list` command to see a list of available models for each framework.
Use the `--use-case` flag to limit the list to models for a particular use case.
```
tlt list models --use-case image_classification
```
```
------------------------------
IMAGE CLASSIFICATION
------------------------------
alexnet (pytorch)
convnext_base (pytorch)
convnext_large (pytorch)
convnext_small (pytorch)
convnext_tiny (pytorch)
densenet121 (pytorch)
densenet161 (pytorch)
densenet169 (pytorch)
densenet201 (pytorch)
efficientnet_b0 (pytorch)
efficientnet_b0 (tensorflow)
efficientnet_b1 (pytorch)
efficientnet_b1 (tensorflow)
efficientnet_b2 (pytorch)
efficientnet_b2 (tensorflow)
efficientnet_b3 (pytorch)
efficientnet_b3 (tensorflow)
efficientnet_b4 (pytorch)
efficientnet_b4 (tensorflow)
efficientnet_b5 (pytorch)
efficientnet_b5 (tensorflow)
efficientnet_b6 (pytorch)
efficientnet_b6 (tensorflow)
efficientnet_b7 (pytorch)
efficientnet_b7 (tensorflow)
efficientnetv2-b0 (tensorflow)
efficientnetv2-b1 (tensorflow)
efficientnetv2-b2 (tensorflow)
efficientnetv2-b3 (tensorflow)
efficientnetv2-s (tensorflow)
googlenet (pytorch)
inception_v3 (tensorflow)
mnasnet0_5 (pytorch)
mnasnet1_0 (pytorch)
mobilenet_v2 (pytorch)
mobilenet_v2_100_224 (tensorflow)
mobilenet_v3_large (pytorch)
...
```

**Train a model**:
For this example, we use the TensorFlow flowers dataset. First, we download and extract the dataset:
```
# Create a directory for the dataset to be downloaded
DATASET_DIR=/tmp/dataset
mkdir -p ${DATASET_DIR}

# Download and extract the dataset
wget -P ${DATASET_DIR} https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar -xzf ${DATASET_DIR}/flower_photos.tgz -C ${DATASET_DIR}

# Set the DATASET_DIR to the extracted images folder
DATASET_DIR=${DATASET_DIR}/flower_photos
```

After the dataset directory is ready, use the `tlt train` command to train one of the models from
`tlt list`. In this example, we use the TensorFlow ResNet50v1.5 model. Make sure to specify
your own file path for the `output-dir`, and the `dataset-dir` should point to the extracted dataset folder.
```
tlt train -f tensorflow --model-name resnet_v1_50 --dataset-dir ${DATASET_DIR} --output-dir /tmp/output
```
```
Model name: resnet_v1_50
Framework: tensorflow
Training epochs: 1
Dataset dir: /tmp/dataset/flower_photos
Output directory: /tmp/output
Found 3670 files belonging to 5 classes.
...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
keras_layer (KerasLayer)    (None, 2048)              23561152
dense (Dense)               (None, 5)                 10245
=================================================================
Total params: 23,571,397
Trainable params: 10,245
Non-trainable params: 23,561,152
_________________________________________________________________
Checkpoint directory: /tmp/output/resnet_v1_50_checkpoints
86/86 [==============================] - 24s 248ms/step - loss: 0.4600 - acc: 0.8438
Saved model directory: /tmp/output/resnet_v1_50/1
```

The `tlt train` command evaluates the model after training completes. The loss and
accuracy values are printed toward the end of the console output, along with the
location where the trained model has been saved.

A trained model can also be evaluated using the `tlt eval` command:
```
tlt eval --model-dir /tmp/output/resnet_v1_50/1 --dataset-dir ${DATASET_DIR}
```

**Benchmark the trained model**:
Benchmark the performance of the trained model using `tlt benchmark`.
Make sure to specify your own file paths for `model-dir` and the `dataset-dir` should point to the extracted dataset folder.
```
tlt benchmark --model-dir /tmp/output/resnet_v1_50/1 --dataset-dir ${DATASET_DIR} --batch-size 512 --mode performance
```
```
Model directory: /tmp/output/resnet_v1_50/1
Dataset directory: /tmp/dataset/flower_photos
Benchmarking mode: performance
Batch size: 512
Model name: resnet_v1_50
Framework: tensorflow
...
performance mode benchmark result:
2022-06-28 10:22:10 [INFO] Batch size = 512
2022-06-28 10:22:10 [INFO] Latency: 3.031 ms
2022-06-28 10:22:10 [INFO] Throughput: 329.878 images/sec
   ```

**Quantize the model**:
Perform post-training quantization using the [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
using the `tlt quantize` command. Make sure to specify your own file paths for `model-dir`, `dataset-dir`, and `output-dir`.
The quantized model will be saved to the output directory.
```
tlt quantize --model-dir /tmp/output/resnet_v1_50/1 --dataset-dir ${DATASET_DIR} --batch-size 512 \
--accuracy-criterion 0.01 --output-dir /tmp/output
```
```
Model directory: /tmp/output/resnet_v1_50/1
Dataset directory: /tmp/dataset/flower_photos
Accuracy criterion: 0.01
Exit policy timeout: 0
Exit policy max trials: 50
Batch size: 512
Output directory: /tmp/output
...
2022-06-28 10:25:58 [INFO] |******Mixed Precision Statistics*****|
2022-06-28 10:25:58 [INFO] +-----------------+----------+--------+
2022-06-28 10:25:58 [INFO] |     Op Type     |  Total   |  INT8  |
2022-06-28 10:25:58 [INFO] +-----------------+----------+--------+
2022-06-28 10:25:58 [INFO] |      Conv2D     |    53    |   53   |
2022-06-28 10:25:58 [INFO] |      MatMul     |    1     |   1    |
2022-06-28 10:25:58 [INFO] |     MaxPool     |    4     |   4    |
2022-06-28 10:25:58 [INFO] |    QuantizeV2   |    5     |   5    |
2022-06-28 10:25:58 [INFO] |    Dequantize   |    4     |   4    |
2022-06-28 10:25:58 [INFO] +-----------------+----------+--------+
2022-06-28 10:25:58 [INFO] Pass quantize model elapsed time: 32164.27 ms
2022-06-28 10:25:58 [INFO] Start to evaluate the TensorFlow model.
2022-06-28 10:26:12 [INFO] Model inference elapsed time: 13921.64 ms
2022-06-28 10:26:12 [INFO] Tune 1 result is: [Accuracy (int8|fp32): 0.9008|0.9022, Duration (seconds) (int8|fp32): 13.9226|17.3321], Best tune result is: [Accuracy: 0.9008, Duration (seconds): 13.9226]
2022-06-28 10:26:12 [INFO] |**********************Tune Result Statistics**********************|
2022-06-28 10:26:12 [INFO] +--------------------+----------+---------------+------------------+
2022-06-28 10:26:12 [INFO] |     Info Type      | Baseline | Tune 1 result | Best tune result |
2022-06-28 10:26:12 [INFO] +--------------------+----------+---------------+------------------+
2022-06-28 10:26:12 [INFO] |      Accuracy      | 0.9022   |    0.9008     |     0.9008       |
2022-06-28 10:26:12 [INFO] | Duration (seconds) | 17.3321  |    13.9226    |     13.9226      |
2022-06-28 10:26:12 [INFO] +--------------------+----------+---------------+------------------+
2022-06-28 10:26:12 [INFO] Save tuning history to /tmp/output/nc_workspace/./history.snapshot.
2022-06-28 10:26:12 [INFO] Specified timeout or max trials is reached! Found a quantized model which meet accuracy goal. Exit.
..
INFO:tensorflow:SavedModel written to: /tmp/output/quantized/resnet_v1_50/1/saved_model.pb
2022-06-28 10:26:13 [INFO] SavedModel written to: /tmp/output/quantized/resnet_v1_50/1/saved_model.pb
2022-06-28 10:26:13 [INFO] Save quantized model to /tmp/output/quantized/resnet_v1_50/1
```

**Benchmark the quantized model**:
The `tlt benchmark` command is used again, but this time the `model-dir` should point
to the quantized model directory.
Make sure to specify your own file paths for `model-dir` and `dataset-dir`. You can then compare
the performance of the full precision model to the quantized model.
```
tlt benchmark --model-dir /tmp/output/quantized/resnet_v1_50/1 --dataset-dir ${DATASET_DIR} --batch-size 512 --mode performance
```
```
Model directory: /tmp/output/quantized/resnet_v1_50/1
Dataset directory: /tmp/dataset/flower_photos
Benchmarking mode: performance
Batch size: 512
Model name: resnet_v1_50
Framework: tensorflow
...
performance mode benchmark result:
2022-06-28 10:28:33 [INFO] Batch size = 512
2022-06-28 10:28:33 [INFO] Latency: 0.946 ms
2022-06-28 10:28:33 [INFO] Throughput: 1056.940 images/sec
```

**Perform graph optimization on the trained model**:
Alternatively, the [Intel Neural Compressor](https://github.com/intel/neural-compressor) can be used to optimize
the full precision graph. Make sure to specify your own file paths for `model-dir` and `output-dir`.
Note that graph optimization is also done as part of the quantization flow, so there is no need to call
`tlt optimize` on a quantized model.
```
tlt optimize --model-dir /tmp/output/resnet_v1_50/1 --output-dir /tmp/output
```
```
Model directory: /tmp/output/resnet_v1_50/1
Model name: resnet_v1_50
Output directory: /tmp/output
Framework: tensorflow
Starting graph optimization
...
2022-06-28 13:50:01 [INFO] Graph optimization is done.
...
2022-06-28 13:51:21 [INFO] SavedModel written to: /tmp/output/optimized/resnet_v1_50/1/saved_model.pb
```

More CLI examples can be found here:
* [Image classification examples](/examples/cli/image_classification.md)
* [Text classification examples](/examples/cli/text_classification.md)
