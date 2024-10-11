import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import pandas as pd
from utils.load_config import load_config
from sklearn.preprocessing import MinMaxScaler
import os

# Load configuration
config = load_config("config.yaml")
dataset_size = config["dataset_size"]
train_size = int(config["train_size_percentage"] * dataset_size)
batch_size = config["batch_size"]

FEATURE_NAMES = ["Depth", "lng", "lat", "bathymetry"]
OUTPUT_NAMES = ["qc", "qt", "fs"]


# Function to get training and test splits
def get_train_and_test_splits(train_size, batch_size=1):
    df = pd.read_csv("data/train.csv")
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(df[FEATURE_NAMES])
    scaled_outputs = output_scaler.fit_transform(df[OUTPUT_NAMES])

    features = {
        name: tf.convert_to_tensor(scaled_features[:, i], dtype=tf.float32)
        for i, name in enumerate(FEATURE_NAMES)
    }
    outputs = tf.convert_to_tensor(scaled_outputs, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((features, outputs))
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset, output_scaler


# Function to create the model inputs
def create_model_inputs():
    inputs = {
        name: layers.Input(name=name, shape=(1,), dtype=tf.float32)
        for name in FEATURE_NAMES
    }
    return inputs


# Define prior and posterior
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )


# Function to create the probabilistic BNN model
def create_probabilistic_bnn_model(hidden_units, activation, train_size):
    inputs = create_model_inputs()
    features = layers.concatenate(list(inputs.values()))

    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / (train_size * 2),
            activation=activation,
        )(features)

    distribution_params = layers.Dense(units=2 * len(OUTPUT_NAMES))(features)
    outputs = tfp.layers.IndependentNormal(len(OUTPUT_NAMES))(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Function to define the loss function
def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


# Function to run the experiment
def run_experiment(model, loss, train_dataset, test_dataset, model_save_dir):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=config["learning_rate"]),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    model_save_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
    )

    print(f"Start training the model in {model_save_dir}...")
    model.fit(
        train_dataset,
        epochs=config["num_epochs"],
        validation_data=test_dataset,
        callbacks=[model_save_callback],
    )
    print("Model training finished.")


# Load data
train_dataset, test_dataset, output_scaler = get_train_and_test_splits(
    train_size, batch_size
)

# Define hyperparameters for tuning
hidden_units_options = [[32], [64], [32, 32]]
activation_options = ["sigmoid", "relu"]  # Added leaky_relu option
optimizer_options = ["adam", "rmsprop"]

# Nested loops for hyperparameter tuning
for optimizer in optimizer_options:
    for hidden_units in hidden_units_options:
        for activation in activation_options:
            model_name = f"opt_{optimizer}_hu_{hidden_units}_act_{activation}"
            model_save_dir = f"models/{model_name}/"
            os.makedirs(model_save_dir, exist_ok=True)

            activation_function = activation

            prob_bnn_model = create_probabilistic_bnn_model(
                hidden_units, activation_function, train_size
            )

            # Choose optimizer based on hyperparameter
            if optimizer == "adam":
                optimizer_instance = keras.optimizers.Adam(
                    learning_rate=config["learning_rate"]
                )
            elif optimizer == "rmsprop":
                optimizer_instance = keras.optimizers.RMSprop(
                    learning_rate=config["learning_rate"]
                )

            run_experiment(
                prob_bnn_model,
                negative_loglikelihood,
                train_dataset,
                test_dataset,
                model_save_dir,
            )


# Function to make predictions using the best model
def make_predictions_and_rescale(model_path, test_dataset, output_scaler, sample=10):
    model = keras.models.load_model(
        model_path, custom_objects={"negative_loglikelihood": negative_loglikelihood}
    )
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

    # Inverse transform the predictions and the CIs for multiple outputs
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
                f"    95% CI: [{round(lower_rescaled[idx][output_idx], 2)} - {round(upper_rescaled[idx][output_idx], 2)}]"
            )
            print(f"    Actual: {round(targets_rescaled[idx][output_idx], 2)}\n")
