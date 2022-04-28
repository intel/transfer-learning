Command Line Interface (CLI)
============================

Use `tlk --help` to see the list of CLI commands. More detailed information on each
command can be found using `tlk <command> --help` (like `tlk train --help`).

Examples
--------

List the available models::

    > tlk list models
    Image Classification
    ----------------------------------------
    resnet_v1_50 (tensorflow)
    resnet_v2_50 (tensorflow)
    resnet_v2_101 (tensorflow)
    mobilenet_v2_100_224 (tensorflow)
    efficientnetv2-s (tensorflow)
    efficientnet_b0 (tensorflow)
    efficientnet_b1 (tensorflow)
    efficientnet_b2 (tensorflow)
    inception_v3 (tensorflow)
    nasnet_large (tensorflow)

Train a model::

    > tlk train -f tensorflow --model-name efficientnet_b0 --dataset-dir /tmp/data --output-dir /tmp/output --dataset-name tf_flowers --dataset-catalog tf_datasets
    Model name: efficientnet_b0
    Framework: tensorflow
    Dataset name: tf_flowers
    Dataset catalog: tf_datasets
    Dataset dir: /tmp/data
    Output directory: /tmp/output
    2022-04-27 10:19:12.651375: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-04-27 10:19:12.658987: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting:

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    keras_layer (KerasLayer)    (None, 1280)              4049564

    dense (Dense)               (None, 5)                 6405

    =================================================================
    Total params: 4,055,969
    Trainable params: 6,405
    Non-trainable params: 4,049,564
    _________________________________________________________________
    None
    86/86 [==============================] - 22s 215ms/step - loss: 0.5106 - acc: 0.8438
    2022-04-27 10:19:38.545222: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    Saved model directory: /tmp/output/efficientnet_b0/1

.. click:: tlk.tools.cli.main:cli_group
   :prog: tlk
   :nested: full