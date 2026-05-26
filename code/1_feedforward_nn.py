# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers, losses, metrics

# Reproducibility: seed Python, NumPy, and TensorFlow in one call
SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()  # forces deterministic GPU/CPU ops

DATA_URL = "https://raw.githubusercontent.com/sekhansen/fra_iese_course_2026/main/code/data/county_vote.csv"

data = pd.read_csv(DATA_URL)
# %%

# plot share of trump_win by decile of population

data["decile"] = pd.qcut(data["log_pop"], 10, labels=False)

data.groupby("decile")["trump_win"].mean().plot()

# plot share of trump_win by decile of white_share

data["decile"] = pd.qcut(data["white_share"], 10, labels=False)

data.groupby("decile")["trump_win"].mean().plot()

# %%

# Prepare input and output variables
x = data[["log_pop", "white_share"]].values  # Convert to NumPy array
y = data["trump_win"].values.reshape(-1, 1)  # Ensure y is a column vector

# %% logistic regression as a one-neuron network

# Define Keras model for logistic regression
logreg = Sequential([
    Input(shape=(2,)),  # Explicit Input layer
    Dense(1, activation="sigmoid")
])

print(logreg.summary())

# Available optimizers
print(dir(optimizers))

# Available loss functions
print(dir(losses))

# Available metrics
print(dir(metrics))

# Compile the model
logreg.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
logreg_history = logreg.fit(x, y, epochs=100, batch_size=50)

# Plot the training history
pd.DataFrame(logreg_history.history).plot()

# print model weights
print(logreg.get_weights())

# %% compare to sklearn logistic regression

logit = LogisticRegression()

logit.fit(x, y)

print(logit.coef_, logit.intercept_)

# %% further training to align estimates

history_cont = logreg.fit(x, y, epochs=500, batch_size=50)

# Plot the training history
pd.DataFrame(history_cont.history).plot()

# print model weights
print(logreg.get_weights())

# %% deep network with two hidden layers

# Standardize inputs so the hidden units train on comparable scales
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = Sequential([
    Input(shape=(2,)),
    Dense(32, activation="relu"),   # Hidden layer with K neurons
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")    # Output layer for binary classification
])

print(model.summary())

# Compile the model
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(x_scaled, y, epochs=500, batch_size=50)

# Plot the training history
pd.DataFrame(history.history).plot()

# %% does the network recover the non-linearity?

# Predicted probability for every county from each trained model
data["nn_prob"] = model.predict(x_scaled).flatten()
data["logreg_prob"] = logreg.predict(x).flatten()

# Within each decile of a covariate, compare the empirical Trump-vote share
# to the average predicted probability from the network and from logistic regression
for cov in ["log_pop", "white_share"]:
    data["decile"] = pd.qcut(data[cov], 10, labels=False)
    means = data.groupby("decile")[["trump_win", "nn_prob", "logreg_prob"]].mean()
    means.plot(title=f"Trump vote: empirical vs network vs logistic, by {cov} decile")

# %% 2D predicted-probability surfaces

# Grid over the observed range of both covariates
lp = np.linspace(data["log_pop"].min(), data["log_pop"].max(), 200)
ws = np.linspace(data["white_share"].min(), data["white_share"].max(), 200)
LP, WS = np.meshgrid(lp, ws)
grid = np.column_stack([LP.ravel(), WS.ravel()])

# Predict on the grid: scale inputs for the network, raw inputs for logistic
nn_surface = model.predict(scaler.transform(grid)).reshape(LP.shape)
logreg_surface = logreg.predict(grid).reshape(LP.shape)

# Side-by-side contour maps with the actual counties overlaid
won = data["trump_win"] == 1
lost = data["trump_win"] == 0

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
for ax, surface, name in zip(axes, [logreg_surface, nn_surface], ["Logistic", "Network"]):
    cf = ax.contourf(LP, WS, surface, levels=20, cmap="Greys", vmin=0, vmax=1)
    ax.scatter(data.loc[lost, "log_pop"], data.loc[lost, "white_share"],
               c="#377eb8", s=12, edgecolor="white", linewidth=0.3, label="Trump lost")
    ax.scatter(data.loc[won, "log_pop"], data.loc[won, "white_share"],
               c="#e41a1c", s=12, edgecolor="white", linewidth=0.3, label="Trump won")
    ax.set_title(f"{name}: P(Trump win)")
    ax.set_xlabel("log_pop")
    ax.set_ylabel("white_share")
    fig.colorbar(cf, ax=ax)

# One shared legend above the panels, clear of all data points
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# %%
