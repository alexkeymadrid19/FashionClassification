# FashionClassification
Using a CNN to classify fashionware using a large a dataset

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "s9Q8-7idZgcR",
        "outputId": "2b44182f-e904-4382-a5ba-224acc209263"
      },
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "TensorFlow: 2.19.0\n",
            "GPU devices: []\n",
            "Train: (60000, 28, 28)  Test: (10000, 28, 28)\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
              "</pre>\n"
            ],
            "text/plain": [
              "\u001b[1mModel: \"sequential\"\u001b[0m\n"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │         <span style=\"color: #00af00; text-decoration-color: #00af00\">1,664</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)     │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)    │       <span style=\"color: #00af00; text-decoration-color: #00af00\">204,928</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)      │       <span style=\"color: #00af00; text-decoration-color: #00af00\">819,456</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2304</span>)           │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)            │       <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>)             │         <span style=\"color: #00af00; text-decoration-color: #00af00\">2,570</span> │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ],
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │         \u001b[38;5;34m1,664\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m64\u001b[0m)     │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m128\u001b[0m)    │       \u001b[38;5;34m204,928\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m128\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m256\u001b[0m)      │       \u001b[38;5;34m819,456\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3\u001b[0m, \u001b[38;5;34m3\u001b[0m, \u001b[38;5;34m256\u001b[0m)      │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2304\u001b[0m)           │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)            │       \u001b[38;5;34m590,080\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m)             │         \u001b[38;5;34m2,570\u001b[0m │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">1,618,698</span> (6.17 MB)\n",
              "</pre>\n"
            ],
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m1,618,698\u001b[0m (6.17 MB)\n"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">1,618,698</span> (6.17 MB)\n",
              "</pre>\n"
            ],
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m1,618,698\u001b[0m (6.17 MB)\n"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ],
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "75/75 - 537s - 7s/step - loss: 3.9173 - sparse_categorical_accuracy: 0.6024 - val_loss: 0.4691 - val_sparse_categorical_accuracy: 0.8315\n",
            "Epoch 2/5\n",
            "75/75 - 587s - 8s/step - loss: 0.4104 - sparse_categorical_accuracy: 0.8490 - val_loss: 0.3828 - val_sparse_categorical_accuracy: 0.8611\n",
            "Epoch 3/5\n",
            "75/75 - 554s - 7s/step - loss: 0.3398 - sparse_categorical_accuracy: 0.8758 - val_loss: 0.3498 - val_sparse_categorical_accuracy: 0.8724\n",
            "Epoch 4/5\n",
            "75/75 - 531s - 7s/step - loss: 0.2981 - sparse_categorical_accuracy: 0.8908 - val_loss: 0.3464 - val_sparse_categorical_accuracy: 0.8758\n",
            "Epoch 5/5\n",
            "75/75 - 529s - 7s/step - loss: 0.2722 - sparse_categorical_accuracy: 0.9002 - val_loss: 0.3236 - val_sparse_categorical_accuracy: 0.8824\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "The filename must end in `.weights.h5`. Received: filepath=model_10x100_weights.h5",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipython-input-5366/4101636240.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m    110\u001b[0m \u001b[0;31m# Save artifacts so you never have to retrain just to get figures again\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    111\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msave\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mMODEL_PATH\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 112\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msave_weights\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mWEIGHTS_PATH\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    113\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Saved model to {MODEL_PATH} and weights to {WEIGHTS_PATH}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    114\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/keras/src/utils/traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    120\u001b[0m             \u001b[0;31m# To get the full stack trace, call:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    121\u001b[0m             \u001b[0;31m# `keras.config.disable_traceback_filtering()`\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 122\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwith_traceback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfiltered_tb\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    123\u001b[0m         \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    124\u001b[0m             \u001b[0;32mdel\u001b[0m \u001b[0mfiltered_tb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/keras/src/saving/saving_api.py\u001b[0m in \u001b[0;36msave_weights\u001b[0;34m(model, filepath, overwrite, max_shard_size, **kwargs)\u001b[0m\n\u001b[1;32m    225\u001b[0m     \u001b[0mfilepath_str\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    226\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mmax_shard_size\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mfilepath_str\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mendswith\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\".weights.h5\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 227\u001b[0;31m         raise ValueError(\n\u001b[0m\u001b[1;32m    228\u001b[0m             \u001b[0;34m\"The filename must end in `.weights.h5`. \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    229\u001b[0m             \u001b[0;34mf\"Received: filepath={filepath_str}\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: The filename must end in `.weights.h5`. Received: filepath=model_10x100_weights.h5"
          ]
        }
      ],
      "source": [
        "# ============================================================\n",
        "# Fashion-MNIST – Original 10×100 Training + Saved Figures/Artifacts\n",
        "# ============================================================\n",
        "\n",
        "# --------- CONFIG ---------\n",
        "EPOCHS = 5\n",
        "STEPS_PER_EPOCH = 75          # preserves your original \"10×100\" setup\n",
        "BATCH_SIZE = 128               # used for evaluation and when Keras batches arrays internally\n",
        "NORMALIZE = False              # keep False to match your original (0–255) setup\n",
        "SEED = 42\n",
        "\n",
        "# Output/Artifact paths\n",
        "FIG_DIR = \"figures\"\n",
        "MODEL_PATH = \"model_10x100.keras\"\n",
        "WEIGHTS_PATH = \"model_10x100_weights.h5\"\n",
        "HISTORY_JSON = \"history_10x100.json\"\n",
        "HISTORY_CSV  = \"training_log_10x100.csv\"\n",
        "\n",
        "# --------- IMPORTS ---------\n",
        "import os, json, random\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import tensorflow as tf\n",
        "\n",
        "from tensorflow.keras.datasets import fashion_mnist\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.callbacks import CSVLogger\n",
        "from sklearn.metrics import confusion_matrix, classification_report\n",
        "\n",
        "# Reproducibility (best effort)\n",
        "random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)\n",
        "\n",
        "print(\"TensorFlow:\", tf.__version__)\n",
        "print(\"GPU devices:\", tf.config.list_physical_devices('GPU'))\n",
        "\n",
        "# --------- PREP OUTPUT DIR ---------\n",
        "os.makedirs(FIG_DIR, exist_ok=True)\n",
        "\n",
        "# ============================================================\n",
        "# 1) DATA\n",
        "# ============================================================\n",
        "(trainX, trainy), (testX, testy) = fashion_mnist.load_data()\n",
        "print('Train:', trainX.shape, ' Test:', testX.shape)\n",
        "\n",
        "# Optional sample grid (saved as figure for your report)\n",
        "plt.figure(figsize=(5,5))\n",
        "for i in range(1, 10):\n",
        "    plt.subplot(3, 3, i)\n",
        "    plt.imshow(trainX[i], cmap='gray')\n",
        "    plt.axis('off')\n",
        "plt.suptitle('Fashion-MNIST Samples', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'sample_grid.png'), dpi=150)\n",
        "plt.close()\n",
        "\n",
        "# Preprocess (match original behavior)\n",
        "if NORMALIZE:\n",
        "    trainX = (trainX.astype('float32') / 255.0)\n",
        "    testX  = (testX.astype('float32')  / 255.0)\n",
        "else:\n",
        "    trainX = trainX.astype('float32')   # 0..255 (original)\n",
        "    testX  = testX.astype('float32')\n",
        "\n",
        "# Add channel dim\n",
        "trainX = np.expand_dims(trainX, -1)     # (N, 28, 28, 1)\n",
        "testX  = np.expand_dims(testX,  -1)\n",
        "\n",
        "# ============================================================\n",
        "# 2) MODEL (your original architecture)\n",
        "# ============================================================\n",
        "def build_model():\n",
        "    m = Sequential()\n",
        "    m.add(Conv2D(64, (5, 5), padding=\"same\", activation=\"relu\", input_shape=(28, 28, 1)))\n",
        "    m.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "    m.add(Conv2D(128, (5, 5), padding=\"same\", activation=\"relu\"))\n",
        "    m.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "    m.add(Conv2D(256, (5, 5), padding=\"same\", activation=\"relu\"))\n",
        "    m.add(MaxPooling2D(pool_size=(2, 2)))\n",
        "    m.add(Flatten())\n",
        "    m.add(Dense(256, activation=\"relu\"))\n",
        "    m.add(Dense(10, activation=\"softmax\"))\n",
        "    return m\n",
        "\n",
        "model = build_model()\n",
        "model.compile(optimizer=Adam(learning_rate=1e-3),\n",
        "              loss='sparse_categorical_crossentropy',\n",
        "              metrics=['sparse_categorical_accuracy'])\n",
        "model.summary()\n",
        "\n",
        "# ============================================================\n",
        "# 3) TRAIN (exactly 10 epochs × 100 steps_per_epoch)\n",
        "#    NOTE: We keep steps_per_epoch=100 to mirror your prior run.\n",
        "#    When passing NumPy arrays, Keras will create batches internally.\n",
        "#    We also log to CSV and JSON so figures can be re-made later.\n",
        "# ============================================================\n",
        "csv_logger = CSVLogger(HISTORY_CSV, append=False)\n",
        "\n",
        "history = model.fit(\n",
        "    trainX, trainy,\n",
        "    epochs=EPOCHS,\n",
        "    steps_per_epoch=STEPS_PER_EPOCH,   # <-- key to preserve your original behavior\n",
        "    validation_split=0.33,\n",
        "    verbose=2,\n",
        "    callbacks=[csv_logger]\n",
        ")\n",
        "\n",
        "# Save artifacts so you never have to retrain just to get figures again\n",
        "model.save(MODEL_PATH)\n",
        "model.save_weights(WEIGHTS_PATH)\n",
        "print(f\"Saved model to {MODEL_PATH} and weights to {WEIGHTS_PATH}\")\n",
        "\n",
        "# Save history to JSON (for re-plotting later without retraining)\n",
        "with open(HISTORY_JSON, \"w\") as f:\n",
        "    json.dump(history.history, f)\n",
        "print(f\"Saved training history to {HISTORY_JSON} and {HISTORY_CSV}\")\n",
        "\n",
        "# ============================================================\n",
        "# 4) LEARNING CURVES (Accuracy & Loss) – saved as PNGs\n",
        "# ============================================================\n",
        "# Accuracy curves\n",
        "plt.figure(figsize=(6,4))\n",
        "plt.plot(history.history['sparse_categorical_accuracy'], label='train')\n",
        "plt.plot(history.history['val_sparse_categorical_accuracy'], label='val')\n",
        "plt.title('Model Accuracy (10×100)')\n",
        "plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'learning_curves_accuracy.png'), dpi=150)\n",
        "plt.close()\n",
        "\n",
        "# Loss curves\n",
        "plt.figure(figsize=(6,4))\n",
        "plt.plot(history.history['loss'], label='train')\n",
        "plt.plot(history.history['val_loss'], label='val')\n",
        "plt.title('Model Loss (10×100)')\n",
        "plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'learning_curves_loss.png'), dpi=150)\n",
        "plt.close()\n",
        "\n",
        "print(\"Saved learning curves to figures/\")\n",
        "\n",
        "# ============================================================\n",
        "# 5) EVALUATION + CONFUSION MATRIX – saved as PNG\n",
        "# ============================================================\n",
        "# Evaluate (just for a test accuracy number in your PDF)\n",
        "test_loss, test_acc = model.evaluate(testX, testy, batch_size=BATCH_SIZE, verbose=0)\n",
        "print(f\"Test Accuracy: {test_acc:.4f}\")\n",
        "\n",
        "# Predictions and confusion matrix\n",
        "y_pred = model.predict(testX, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)\n",
        "\n",
        "print(\"\\nClassification report (test):\\n\")\n",
        "try:\n",
        "    from sklearn.metrics import classification_report\n",
        "    print(classification_report(testy, y_pred, digits=4))\n",
        "except Exception as e:\n",
        "    print(\"Classification report skipped:\", e)\n",
        "\n",
        "cm = confusion_matrix(testy, y_pred)\n",
        "plt.figure(figsize=(7,6))\n",
        "sns.heatmap(cm, cmap='Blues', annot=False, fmt='d')\n",
        "plt.title('Confusion Matrix – Fashion-MNIST (Test) [10×100]')\n",
        "plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=150)\n",
        "plt.close()\n",
        "\n",
        "print(\"Saved confusion matrix to figures/confusion_matrix.png\")\n",
        "\n",
        "# ============================================================\n",
        "# 6) (OPTIONAL) ZIP FIGURES FOR EASY DOWNLOAD\n",
        "# ============================================================\n",
        "# Uncomment to create a single ZIP with all images:\n",
        "# !zip -r figures.zip figures\n",
        "# After it finishes, download figures.zip from the Files pane (left sidebar).\n"
      ]
    }
  ]
}
