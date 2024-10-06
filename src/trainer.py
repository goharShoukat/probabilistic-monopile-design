import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import pandas as pd
from utils.load_config import load_config
from sklearn.preprocessing import MinMaxScaler
import os

config = load_config("config.yaml")
hidden_units = config["hidden_units"]
learning_rate = config["learning_rate"]
dataset_size = config["dataset_size"]
train_size = int(config["train_size_percentage"] * dataset_size)
batch_size = config["batch_size"]


FEATURE_NAMES = ["Depth", "lng", "lat", "bathymetry"]
OUTPUT_NAMES = [
    "qc",
    "qt",
    "fs",
]


def get_train_and_test_splits(train_size, batch_size=1):
    df = pd.read_csv("data/train.csv")
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(df[FEATURE_NAMES])
    scaled_outputs = output_scaler.fit_transform(df[OUTPUT_NAMES])

    df_features = pd.DataFrame(scaled_features, columns=FEATURE_NAMES)
    df_outputs = pd.DataFrame(scaled_outputs, columns=OUTPUT_NAMES)

    features = {
        name: tf.convert_to_tensor(df_features[name].values, dtype=tf.float32)
        for name in FEATURE_NAMES
    }
    outputs = tf.convert_to_tensor(df_outputs[OUTPUT_NAMES].values, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((features, outputs))
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset, output_scaler


def run_experiment(model, loss, train_dataset, test_dataset):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    model_save_dir = "models/depth-2/"
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_dir,
        monitor="loss",
        mode="min",
        save_best_only=True,
        period=2,
    )
    print("Start training the model...")
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
        # callbacks=[model_save_callback],
    )
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs


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


train_dataset, test_dataset, output_scalar = get_train_and_test_splits(
    train_size, batch_size
)
num_epochs = 100
mse_loss = keras.losses.MeanSquaredError()
# train_sample_size = int(train_size * 0.3)
# small_train_dataset = train_dataset.unbatch().take(train_sample_size).batch(batch_size)


def create_probablistic_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / (train_size * 2),
            activation="relu",
        )(features)

    distribution_params = layers.Dense(units=2 * len(OUTPUT_NAMES))(features)
    outputs = tfp.layers.IndependentNormal(len(OUTPUT_NAMES))(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


num_epochs = 5
prob_bnn_model = create_probablistic_bnn_model(train_size)
run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)


def make_predictions_and_rescale(model, test_dataset, output_scaler, sample=10):
    examples, targets = list(
        test_dataset.unbatch().shuffle(batch_size * 10).batch(sample)
    )[0]
    prediction_distribution = model(examples)

    # Extract means and stddevs for multiple outputs
    prediction_mean = prediction_distribution.mean().numpy()
    prediction_stdv = prediction_distribution.stddev().numpy()

    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = prediction_mean + (1.96 * prediction_stdv)
    lower = prediction_mean - (1.96 * prediction_stdv)

    # Step 3: Inverse transform the predictions and the CIs for multiple outputs
    prediction_mean_rescaled = output_scaler.inverse_transform(prediction_mean)
    upper_rescaled = output_scaler.inverse_transform(upper)
    lower_rescaled = output_scaler.inverse_transform(lower)
    targets_rescaled = output_scaler.inverse_transform(targets.numpy())

    # Print the rescaled predictions and actual values
    for idx in range(sample):
        print(f"Sample {idx + 1}:")
        for output_idx, output_name in enumerate(OUTPUT_NAMES):
            print(f"  Output {output_name}:")
            print(
                f"    Prediction mean: {round(prediction_mean_rescaled[idx][output_idx], 2)}"
            )
            print(
                f"    95% CI: [{round(upper_rescaled[idx][output_idx], 2)} - {round(lower_rescaled[idx][output_idx], 2)}]"
            )
            print(f"    Actual: {round(targets_rescaled[idx][output_idx], 2)}\n")


make_predictions_and_rescale(prob_bnn_model, test_dataset, output_scalar)
