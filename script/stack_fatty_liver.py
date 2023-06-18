import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

os.environ["PATH_DIR_RESULTS"] = "/home/adam/sail-model-optimizer/script/flf/results"

df_output = pd.read_csv("script/fatty_liver_output.csv")
df_input = pd.read_csv("script/fatty_liver_input.csv")

df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(
    df_input,
    df_output,
    test_size=0.1,
    random_state=42,
)


# Logistic Regression
path_dir_results = os.environ["PATH_DIR_RESULTS"]
path_file_results = os.path.join(path_dir_results, f"logistic.json")
dict_run = {}

try:
    print("Reading dict_run logistic")
    with open(path_file_results, "r") as file_model:
        dict_run = json.load(file_model)
except IOError:
    print("Error reading the file!")

list_feature_selected_lr = dict_run["list_feature_selected"]
dict_params = dict_run["dict_params_current"]
array_input_train_lr = df_input_train[list_feature_selected_lr].to_numpy()
array_output_train = df_output_train.to_numpy()
array_input_test_lr = df_input_test[list_feature_selected_lr].to_numpy()
array_output_test_true = df_output_test.to_numpy()

lr_model = LogisticRegression()
lr_model.set_params(**dict_params)
lr_model.fit(array_input_train_lr, array_output_train)


# XGBoost
path_dir_results = os.environ["PATH_DIR_RESULTS"]
path_file_results = os.path.join(path_dir_results, f"xgboost.json")
dict_run = {}

try:
    print("Reading dict_run xgboost")
    with open(path_file_results, "r") as file_model:
        dict_run = json.load(file_model)
except IOError:
    print("Error reading the file!")

list_feature_selected_xgb = dict_run["list_feature_selected"]
dict_params = dict_run["dict_params_current"]
array_input_train_xgb = df_input_train[list_feature_selected_xgb].to_numpy()
array_input_test_xgb = df_input_test[list_feature_selected_xgb].to_numpy()

xgb_model = xgb.XGBClassifier()
xgb_model.set_params(**dict_params)
xgb_model.fit(array_input_train_xgb, array_output_train)

# Bayesian Network
path_dir_results = os.environ["PATH_DIR_RESULTS"]
path_file_results = os.path.join(path_dir_results, "bayesian_network.json")
dict_run = {}

try:
    print("Reading dict_run bayesian")
    with open(path_file_results, "r") as file_model:
        dict_run = json.load(file_model)
except IOError:
    print("Error reading the file!")

list_feature_selected_bayesian = dict_run["list_feature_selected"]
dict_params = dict_run["dict_params_current"]
array_input_train_bayesian = df_input_train[list_feature_selected_bayesian].to_numpy()
array_input_test_bayesian = df_input_test[list_feature_selected_bayesian].to_numpy()

bayesian_model = GaussianNB()
bayesian_model.set_params(**dict_params)
bayesian_model.fit(array_input_train_bayesian, array_output_train)

# Neural Network

try:
    print("Reading dict_run neural")
    with open(path_file_results, "r") as file_model:
        dict_run = json.load(file_model)
except IOError:
    print("Error reading the file!")

list_feature_selected_neural = dict_run["list_feature_selected"]
dict_params = dict_run["dict_params_current"]
array_input_train_neural = df_input_train[list_feature_selected_neural].to_numpy()
array_input_test_neural = df_input_test[list_feature_selected_neural].to_numpy()

# Set the random seed
random_seed = 42
np.random.seed(random_seed)

# Create and train the Neural Network model
neural_model = MLPClassifier(random_state=random_seed)
neural_model.set_params(**dict_params)
neural_model.fit(array_input_train_neural, array_output_train)


# Predict probabilities using trained models
xgb_probs = xgb_model.predict_proba(array_input_test_xgb)
lr_probs = lr_model.predict_proba(array_input_test_lr)
bayesian_probs = bayesian_model.predict_proba(array_input_test_bayesian)
neural_probs = neural_model.predict_proba(array_input_test_neural)

# Combine probabilities using average
combined_probs = (xgb_probs + lr_probs + bayesian_probs + neural_probs) / 4

# Calculate fpr and tpr for each model
fpr_meta, tpr_meta, _ = roc_curve(df_output_test, combined_probs[:, 1])
fpr_xgb, tpr_xgb, _ = roc_curve(df_output_test, xgb_probs[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(df_output_test, lr_probs[:, 1])
fpr_bayesian, tpr_bayesian, _ = roc_curve(df_output_test, bayesian_probs[:, 1])
fpr_neural, tpr_neural, _ = roc_curve(df_output_test, neural_probs[:, 1])

# Calculate AUROC for each model
auroc_meta = roc_auc_score(df_output_test, combined_probs[:, 1])
auroc_xgb = roc_auc_score(df_output_test, xgb_probs[:, 1])
auroc_lr = roc_auc_score(df_output_test, lr_probs[:, 1])
auroc_bayesian = roc_auc_score(df_output_test, bayesian_probs[:, 1])
auroc_neural = roc_auc_score(df_output_test, neural_probs[:, 1])

# Set Seaborn style
sns.set(style="whitegrid")

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot ROC curves with custom colors
ax.plot(
    fpr_xgb, tpr_xgb, label=f"XGBoost Model (AUROC = {auroc_xgb:.3f})", linewidth=2, color="#3498DB", linestyle="dotted"
)
ax.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression Model (AUROC = {auroc_lr:.3f})",
    linewidth=2,
    color="#2ECC71",
    linestyle="dotted",
)
ax.plot(
    fpr_neural,
    tpr_neural,
    label=f"Neural Network (AUROC = {auroc_neural:.3f})",
    linewidth=2,
    color="#E74C3C",
    linestyle="dotted",
)
ax.plot(
    fpr_bayesian,
    tpr_bayesian,
    label=f"Bayesian Network Model (AUROC = {auroc_bayesian:.3f})",
    linewidth=2,
    color="#F1C40F",
    linestyle="dotted",
)
ax.plot(fpr_meta, tpr_meta, label=f"Meta Model (AUROC = {auroc_meta:.3f})", linewidth=2, color="#555555")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, linestyle="dashed", alpha=0.7)  # Diagonal line for reference

# Set labels and title with custom font properties
font_family = "Roboto"  # Replace with the font you chose
font_properties = {"fontname": font_family, "fontsize": 12, "fontweight": "bold"}
ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")

# Set legend with custom font properties and shadow
ax.legend(loc="lower right", fontsize=10, shadow=True)

# Save the plot to a file with high DPI for better resolution
plt.savefig("roc_curves.png", dpi=300)

plt.close()


# Define the two lists of features used by the classifiers
features1 = list_feature_selected_lr
features2 = list_feature_selected_xgb
features3 = list_feature_selected_bayesian
features4 = list_feature_selected_neural


# Convert the feature lists to sets
features1_set = set(features1)
features2_set = set(features2)
features3_set = set(features3)
features4_set = set(features4)

# Create the Venn diagrams
venn12 = venn2([features1_set, features2_set], set_labels=("LR", "XGB"))
venn34 = venn2([features3_set, features4_set], set_labels=("Bayesian", "Neural"))

# Customize the circles in the diagrams
venn2_circles([features1_set, features2_set])
venn2_circles([features3_set, features4_set])

# Set the title
plt.title("Venn Diagram of Feature Sets")

# Save the Venn diagram to a file with high DPI for better resolution
plt.savefig("subclassifier_feature_comparison.png", dpi=300)

plt.close()


features = (
    list_feature_selected_lr + list_feature_selected_xgb + list_feature_selected_bayesian + list_feature_selected_neural
)

# Count the occurrence of each feature
feature_counts = {}
for feature in features:
    feature_counts[feature] = feature_counts.get(feature, 0) + 1

# Normalize the occurrence counts
counts = np.array(list(feature_counts.values()))
normalized_counts = np.log10(counts)

# Generate the word cloud with feature occurrences as size
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(feature_counts)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Display the word cloud with font size based on normalized occurrence
ax.imshow(wordcloud, interpolation="bilinear")
ax.set_axis_off()

# Set the font sizes based on normalized feature occurrences
for word, count in feature_counts.items():
    size = normalized_counts[count] * 30  # Adjust the scaling factor as needed
    x, y = np.random.randint(0, 800), np.random.randint(0, 400)
    ax.text(x, y, word, fontsize=size, color="black", ha="center", va="center")  # x position  # y position

# Add a title
ax.set_title("Feature Occurrence Word Cloud", fontsize=14, fontweight="bold")

# Save the word cloud to a file
plt.savefig("feature_occurrence_wordcloud.png", dpi=300)
