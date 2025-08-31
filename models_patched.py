import tensorflow as tf


def build_model_b_patched(input_shape, num_classes):
    """
    Model B (patched): Residual regularized CNN.
    - Conv-BN-ReLU x2 per stage with residual add (+ channel match if needed)
    - MaxPool + SpatialDropout2D per stage
    - GAP + BN + Dense(128, ReLU) head
    - L2 regularization on conv kernels
    """
    wd = 1e-4
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for f in [32, 64, 128]:
        shortcut = x
        x = tf.keras.layers.Conv2D(
            f, 3, padding="same", use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(wd),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(
            f, 3, padding="same", use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(wd),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if shortcut.shape[-1] != f:
            shortcut = tf.keras.layers.Conv2D(
                f, 1, padding="same", use_bias=False,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(wd),
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(0.10)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ModelB_ResidualReg")


def build_model_c_patched(input_shape, num_classes):
    """
    Model C (patched): Depthwise-Separable v2 with improved head.
    - Two SepConv blocks per stage (48, 96), BN, ReLU
    - MaxPool + SpatialDropout2D per stage
    - GAP + BN + Dense(128, ReLU) head
    - L2 regularization on depthwise/pointwise kernels
    """
    wd = 1e-4
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for f in [48, 96]:
        x = tf.keras.layers.SeparableConv2D(
            f, 3, padding="same", use_bias=False,
            depthwise_regularizer=tf.keras.regularizers.l2(wd),
            pointwise_regularizer=tf.keras.regularizers.l2(wd),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.SeparableConv2D(
            f, 3, padding="same", use_bias=False,
            depthwise_regularizer=tf.keras.regularizers.l2(wd),
            pointwise_regularizer=tf.keras.regularizers.l2(wd),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(0.15)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ModelC_SeparableV2")


def build_model_e(input_shape, num_classes):
    """
    Model E: Transfer learning with MobileNetV2.
    - Rescale [0,1] -> [-1,1] to match MobileNetV2 preprocessing
    - Base frozen; light head on top
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ModelE_MobileNetV2")

