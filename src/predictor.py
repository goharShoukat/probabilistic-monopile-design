# import glob
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_probability as tfp
# import pandas as pd
# from utils.load_config import load_config
# from sklearn.preprocessing import MinMaxScaler
# import os

# FEATURE_NAMES = ["Depth", "lng", "lat", "bathymetry"]
# OUTPUT_NAMES = ["qc", "qt", "qt"]
# config = load_config("config.yaml")
# dataset_size = config["dataset_size"]
# train_size = int(config["train_size_percentage"] * dataset_size)
# batch_size = config["batch_size"]


# # Function to define the loss function
# def negative_loglikelihood(targets, estimated_distribution):
#     return -estimated_distribution.log_prob(targets)


# def get_train_and_test_splits(train_size, batch_size=1):
#     df = pd.read_csv("data/train.csv")
#     output_scaler = MinMaxScaler(feature_range=(0, 1))
#     feature_scaler = MinMaxScaler(feature_range=(0, 1))

#     scaled_features = feature_scaler.fit_transform(df[FEATURE_NAMES])
#     scaled_outputs = output_scaler.fit_transform(df[OUTPUT_NAMES])

#     features = {
#         name: tf.convert_to_tensor(scaled_features[:, i], dtype=tf.float32)
#         for i, name in enumerate(FEATURE_NAMES)
#     }
#     outputs = tf.convert_to_tensor(scaled_outputs, dtype=tf.float32)

#     dataset = tf.data.Dataset.from_tensor_slices((features, outputs))
#     train_dataset = (
#         dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
#     )
#     test_dataset = dataset.skip(train_size).batch(batch_size)

#     return train_dataset, test_dataset, output_scaler


# def make_predictions_from_all_models(
#     models_dir, test_dataset, output_scaler, sample=10
# ):
#     # Get a list of all the model directories
#     # model_dirs = glob.glob(os.path.join(models_dir, "*"))
#     # model_dirs = ["opt_adamhu_32_act_relu", "opt_adamhu_32_act_sigmoid"]
#     model_dirs = [
#         "opt_adamhu_64_act_relu",
#     ]

#     # Loop through each model directory
#     for model_dir in model_dirs:
#         print(model_dir)
#         model_path = model_dir

#         if os.path.exists(model_path):
#             print(f"Making predictions for model: {model_dir}")
#             # Load the model
#             model = keras.models.load_model(
#                 model_path,
#                 custom_objects={"negative_loglikelihood": negative_loglikelihood},
#             )

#             # Get a sample from the test dataset
#             examples, targets = list(
#                 test_dataset.unbatch().shuffle(batch_size * 10).batch(sample)
#             )[0]

#             # Make predictions
#             prediction_distribution = model(examples)

#             # Extract means and stddevs for multiple outputs
#             prediction_mean = prediction_distribution.mean().numpy()
#             prediction_stdv = prediction_distribution.stddev().numpy()

#             # The 95% CI is computed as mean ± (1.96 * stdv)
#             upper = prediction_mean + (1.96 * prediction_stdv)
#             lower = prediction_mean - (1.96 * prediction_stdv)

#             # Inverse transform the predictions and the CIs for multiple outputs
#             prediction_mean_rescaled = output_scaler.inverse_transform(prediction_mean)
#             upper_rescaled = output_scaler.inverse_transform(upper)
#             lower_rescaled = output_scaler.inverse_transform(lower)
#             targets_rescaled = output_scaler.inverse_transform(targets.numpy())

#             # Print the rescaled predictions and actual values
#             print(f"Model {model_dir}:")
#             for idx in range(sample):
#                 print(f"Sample {idx + 1}:")
#                 for output_idx, output_name in enumerate(OUTPUT_NAMES):
#                     print(f"  Output {output_name}:")
#                     print(
#                         f"    Prediction mean: {round(prediction_mean_rescaled[idx][output_idx], 2)}"
#                     )
#                     print(
#                         f"    95% CI: [{round(lower_rescaled[idx][output_idx], 2)} - {round(upper_rescaled[idx][output_idx], 2)}]"
#                     )
#                     print(
#                         f"    Actual: {round(targets_rescaled[idx][output_idx], 2)}\n"
#                     )
#             print("\n" + "-" * 50 + "\n")
#         else:
#             print(f"No model found in {model_dir}")
#         break


# # Call the function to load all models and make predictions
# models_directory = "models"  # Directory where all models are stored
# _, test_dataset, output_scaler = get_train_and_test_splits(train_size, batch_size)
# make_predictions_from_all_models(models_directory, test_dataset, output_scaler)

# %%
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
from sklearn.metrics import mean_squared_error

hidden_units = [8, 8]
learning_rate = 0.0001
batch_size = 64
num_epochs = 100
dataset_size = 26900
train_size_percentage = 0.90
train_size = int(train_size_percentage * dataset_size)
FEATURE_NAMES = ["Depth", "lng", "lat", "bathymetry"]
OUTPUT_NAMES = ["Smooth qt"]


train = pd.read_csv("train2.csv")
train = train.astype("float64")
X = np.array(train[["Depth", "lat", "lng", "bathymetry"]].copy())
Y = np.array(train["Smooth qt"].copy())
X, Y = shuffle(X, Y)

scalarX = MinMaxScaler(feature_range=(0, 1))
X = scalarX.fit_transform(X)

# scalarY = MinMaxScaler(feature_range=(0, 1))
# Y = scalarY.fit_transform(Y)

X1 = X[:, 0]  # depth
X2 = X[:, 1]  # lat
X3 = X[:, 2]  # lng
X4 = X[:, 3]  # bathy

Y1 = Y  # qc


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


models = os.listdir("models/bnn_prob_model/")
error = {}
# Load the model with the custom loss function
# for model_path in models:
for model_path in models:
    print(model_path)
    with keras.utils.custom_object_scope(
        {"negative_loglikelihood": negative_loglikelihood}
    ):
        model = keras.models.load_model("models/bnn_prob_model/" + model_path)

    results = model({"depth": X1, "lat": X2, "lng": X3, "bathymetry": X4})
    results_mean = results.mean().numpy().tolist()
    prediction_stdv = results.stddev().numpy()

    upper = (results_mean + (1.96 * prediction_stdv)).tolist()
    lower = (results_mean - (1.96 * prediction_stdv)).tolist()

    for idx in range(len(train)):
        print(
            f"Prediction mean: {round(results_mean[idx][0], 2)}, "
            f"stddev: {round(prediction_stdv[idx][0], 2)}, "
            f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
            f" - Actual: {Y[idx]}"
        )
    print("-------------------------------------------------------------")
    error[model_path] = mean_squared_error(Y, results_mean)
print(error.items())

print(min(error))
