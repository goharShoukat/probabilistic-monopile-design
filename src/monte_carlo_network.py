import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import pandas as pd
from utils.load_config import load_config
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.utils import shuffle

feat = "qt"
learning_rate = 0.0001
batch_size = 64

FEATURE_NAMES = ["Depth", "lng", "lat", "bathymetry"]
OUTPUT_NAMES = [f"Smooth {feat}"]

hidden_unit_layers = [
    [8],
    [16],
    [32],
    [64],
    [8, 8],
    [8, 16],
    [8, 32],
    [8, 64],
    [16, 8],
    [16, 16],
    [16, 32],
    # [16, 64],
    # [32, 8],
    # [32, 32],
    # [32, 64],
    # [64, 8],
    # [64, 16],
    # [64, 32],
    # [64, 64],
]
kl_weights = [
    0.01,
    0.001,
    0.001,
    0.0001,
    0.02,
    0.002,
    0.0002,
    0.0125,
    0.00125,
    0.000125,
    0.025,
    0.0025,
    0.0025,
    0.05,
    0.005,
    0.0005,
]

for kl_weight in kl_weights:
    for layer in hidden_unit_layers:
        train = pd.read_csv("train2.csv")
        train = train.astype("float64")
        X = np.array(train[["Depth", "lat", "lng", "bathymetry"]].copy())
        Y = np.array(train[f"Smooth {feat}"].copy())
        X, Y = shuffle(X, Y)

        scalarX = MinMaxScaler(feature_range=(0, 1))
        X = scalarX.fit_transform(X)

        # scalarY = MinMaxScaler(feature_range=(0, 1))
        # Y = scalarY.fit_transform(Y)

        X1 = X[:, 0]  # depth
        X2 = X[:, 1]  # lat
        X3 = X[:, 2]  # lng
        X4 = X[:, 3]  # bathy

        Y1 = Y  # Smooth qt

        input = {"depth": X1, "lat": X2, "lng": X3, "bathymetry": X4}
        output = {f"Smooth {feat}": Y1}

        input1 = keras.Input(shape=(1,), name="depth")
        input2 = keras.Input(shape=(1,), name="lat")
        input3 = keras.Input(shape=(1,), name="lng")
        input4 = keras.Input(shape=(1,), name="bathymetry")
        inputs = [input1, input2, input3, input4]

        def prior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = keras.Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.MultivariateNormalDiag(
                            loc=tf.zeros(n), scale_diag=tf.ones(n)
                        )
                    )
                ]
            )
            return prior_model

        def posterior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            posterior_model = keras.Sequential(
                [
                    tfp.layers.VariableLayer(
                        tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
                    ),
                    tfp.layers.MultivariateNormalTriL(n),
                ]
            )
            return posterior_model

        features = keras.layers.Concatenate(axis=1)(inputs)
        checkpoint_path = (
            f"models/monte-carlo-method/{feat}-kl-weight-{kl_weight}-neurons"
        )
        for units in layer:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=kl_weight,
                activation="sigmoid",
            )(features)
            checkpoint_path += "-" + str(units)
        # The output is deterministic: a single point estimate.
        outputs = layers.Dense(units=1)(features)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

        model_save_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_root_mean_squared_error",
            mode="min",
            save_best_only=True,
            period=1,
        )
        print("Start training the model...")
        model.fit(
            {"depth": X1, "lat": X2, "lng": X3, "bathymetry": X4},
            Y1,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=500,
            verbose=1,
            shuffle=True,
            callbacks=[model_save_callback],
        )
        # testX = {
        #     "depth": X1[500:600],
        #     "lat": X2[500:600],
        #     "lng": X3[500:600],
        #     "bathymetry": X4[500:600],
        # }
        # testY = Y1[500:600]

        # predicted = []
        # for _ in range(100):
        #     predicted.append(model(testX).numpy())
        # predicted = np.concatenate(predicted, axis=1)

        # prediction_mean = np.mean(predicted, axis=1).tolist()
        # prediction_min = np.min(predicted, axis=1).tolist()
        # prediction_max = np.max(predicted, axis=1).tolist()
        # prediction_range = (
        #     np.max(predicted, axis=1) - np.min(predicted, axis=1)
        # ).tolist()

        # for idx in range(100):
        #     print(
        #         f"Predictions mean: {round(prediction_mean[idx], 2)}, "
        #         f"min: {round(prediction_min[idx], 2)}, "
        #         f"max: {round(prediction_max[idx], 2)}, "
        #         f"range: {round(prediction_range[idx], 2)} - "
        #         f"Actual: {testY[idx]}"
        #     )
