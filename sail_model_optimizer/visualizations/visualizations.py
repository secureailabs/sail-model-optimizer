import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib_venn import venn3
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

# TODO: load model from file
# def load_model_from_file(path_file_results, model_class, array_input_train, array_output_train):
# TODO: load model from file
# dict_run = {}
# try:
#     print(f"Reading dict_run {model_class.__name__.lower()}")
#     with open(path_file_results, "r") as file_model:
#         dict_run = json.load(file_model)
# except IOError:
#     print("Error reading the file!")

# list_feature_selected_model = dict_run["list_feature_selected"]
# dict_params = dict_run["dict_params_current"]
# array_input_train_model = array_input_train[list_feature_selected_model].to_numpy()

# model = model_class()
# model.set_params(**dict_params)
# model.fit(array_input_train_model, array_output_train)

# return model, list_feature_selected_model

# TODO: load model from file and save
# def train_model_from_file_save(path_file_results, model_class, array_input_train, array_output_train):
#     dict_run = {}
#     try:
#         print(f"Reading dict_run {model_class.__name__.lower()}")
#         with open(path_file_results, "r") as file_model:
#             dict_run = json.load(file_model)
#     except IOError:
#         print("Error reading the file!")

#     list_feature_selected_model = dict_run["list_feature_selected"]
#     dict_params = dict_run["dict_params_current"]
#     array_input_train_model = array_input_train[list_feature_selected_model].to_numpy()

#     model = model_class()
#     model.set_params(**dict_params)
#     model.fit(array_input_train_model, array_output_train)

#     return model, list_feature_selected_model


def train_model_from_file(path_file_results, model_class, array_input_train, array_output_train):
    dict_run = {}
    try:
        print(f"Reading dict_run {model_class.__name__.lower()}")
        with open(path_file_results, "r") as file_model:
            dict_run = json.load(file_model)
    except IOError:
        print("Error reading the file!")

    list_feature_selected_model = dict_run["list_feature_selected"]
    dict_params = dict_run["dict_params_current"]
    array_input_train_model = array_input_train[list_feature_selected_model].to_numpy()

    model = model_class()
    model.set_params(**dict_params)
    model.fit(array_input_train_model, array_output_train)

    return model, list_feature_selected_model


def calculate_results(model_data, df_output_test):
    results = {}
    combined_probs = []

    for model, input_data, model_name in model_data:
        probs = model.predict_proba(input_data)
        fpr, tpr, _ = roc_curve(df_output_test, probs[:, 1])
        auroc = roc_auc_score(df_output_test, probs[:, 1])

        results[model_name] = {"fpr": fpr, "tpr": tpr, "auroc": auroc}

        if model_name != "Meta":
            combined_probs.append(probs)

    # Calculate combined probabilities using voting
    combined_probs = np.mean(combined_probs, axis=0)
    fpr_meta, tpr_meta, _ = roc_curve(df_output_test, combined_probs[:, 1])
    auroc_meta = roc_auc_score(df_output_test, combined_probs[:, 1])

    results["Meta"] = {"fpr": fpr_meta, "tpr": tpr_meta, "auroc": auroc_meta}

    return results


def plot_roc_curves(results):
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model_results in results.items():
        if model_name == "Combined":
            continue

        fpr = model_results["fpr"]
        tpr = model_results["tpr"]
        auroc = model_results["auroc"]

        label = f"{model_name} Model (AUROC = {auroc:.3f})"

        if model_name == "Meta":

            ax.plot(fpr, tpr, label=label, linewidth=4, linestyle="dotted")
        else:
            ax.plot(fpr, tpr, label=label, linewidth=2, linestyle="dotted")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7)  # Diagonal line for reference

    # Set labels and title with custom font properties
    font_family = "Roboto"  # Replace with the font you chose
    font_properties = {"fontname": font_family, "fontsize": 12, "fontweight": "bold"}
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")

    # Set legend with custom font properties and shadow
    ax.legend(loc="lower right", fontsize=10, shadow=True)

    # Save the plot to a file with high DPI for better resolution
    # plt.savefig(os.environ["PATH_DIR_RESULTS"] + "/visualizations/roc_curves.png", dpi=300)
    # plt.close()

    ## Render the figure to a response
    # buf = io.BytesIO()
    # fig.savefig(buf, format="png")
    # buf.seek(0)
    # print(buf)

    return fig


def plot_feature_venn(feature_lists, set_labels=("Set 1", "Set 2", "Set 3")):
    # Convert the feature lists to sets
    feature_sets = [set(features) for features in feature_lists]

    # Create the Venn diagrams
    venn = venn3(feature_sets, set_labels=set_labels)

    # Set the title
    plt.title("Venn Diagram of Feature Sets")

    # Save the Venn diagram to a file with high DPI for better resolution
    # plt.savefig(os.environ["PATH_DIR_RESULTS"] + "/visualizations/subclassifier_feature_comparison.png", dpi=300)
    return plt


def generate_feature_wordcloud(feature_lists):
    # Combine all feature lists into a single list
    output_file = os.environ["PATH_DIR_RESULTS"] + "/visualizations/feature_wordcloud.png"
    all_features = []
    for feature_list in feature_lists:
        all_features += feature_list

    # Create a string containing all the features
    features_text = " ".join(all_features)

    # Generate the word cloud with fancy color scheme and frequency-based sizes
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Spectral", prefer_horizontal=0.9, min_font_size=8
    ).generate_from_text(features_text)

    # Display the word cloud with frequency-based sizes
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    # Save the word cloud to a file
    wordcloud.to_file(output_file)

    # Show the word cloud
    plt.show()


def generate_feature_frequency_graph(feature_lists):
    # Flatten the list of feature lists
    flat_feature_lists = [feature for sublist in feature_lists for feature in sublist]

    # Count the occurrence of each feature
    feature_counts = {feature: flat_feature_lists.count(feature) for feature in set(flat_feature_lists)}

    # Sort the features by frequency in descending order
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # Extract the feature names and counts for plotting
    features = [f[0] for f in sorted_features]
    counts = [f[1] for f in sorted_features]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get the number of unique lists
    num_lists = len(feature_lists)

    # Set the color palette based on the number of lists
    color_palette = colors.cm.get_cmap("viridis", num_lists)

    # Determine the width and offset for each bar group
    bar_width = 0.8 / num_lists
    offset = np.linspace(-0.4 + 0.5 * bar_width, 0.4 - 0.5 * bar_width, num_lists)

    # Plot the bar graph with colored bars based on the list the features came from
    for i, feature_list in enumerate(feature_lists):
        feature_indices = [features.index(feature) for feature in feature_list]
        ax.bar(
            np.arange(len(feature_indices)) + offset[i],
            [counts[idx] for idx in feature_indices],
            width=bar_width,
            color=color_palette(i),
            edgecolor="black",
        )

    # Set labels and title with increased font size and font weight
    ax.set_xlabel("Features", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax.set_title("Feature Frequency", fontsize=16, fontweight="bold")

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", labels=features)

    # Create a legend for the color-coded bars
    ax.legend(labels=["List {}".format(i + 1) for i in range(num_lists)], loc="upper right", fontsize=10)

    # Adjust the layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the graph to a file with high DPI for better resolution
    plt.savefig(os.environ["PATH_DIR_RESULTS"] + "/visualizations/feature_frequency_graph.png", dpi=300)
    plt.close()


def visualize_pca(df_features, df_target, feature_lists, n_components=4, save_path="PCA"):
    unique_features = list(set([feature for sublist in feature_lists for feature in sublist]))

    save_path = os.environ["PATH_DIR_RESULTS"] + "/visualizations/" + save_path
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_features[unique_features])

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_scaled)

    # Create a DataFrame with the principal components and target variable
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    principal_df = pd.DataFrame(data=principal_components, columns=pc_columns)
    principal_df["target"] = df_target

    # Compute explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    explained_variance_pc = [round(variance * 100, 2) for variance in explained_variance]
    explained_variance_text = ", ".join(
        [f"PC{i+1} Explained Variance: {explained_variance_pc[i]}%" for i in range(n_components)]
    )
    title = f"PCA Visualization of Features ({n_components} components)"

    # Get feature importance by sorting the loadings for each principal component
    feature_importance = np.abs(pca.components_)
    feature_names = unique_features

    # Plot PC4 with shading as the 4th dimension
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        principal_df[pc_columns[0]],
        principal_df[pc_columns[1]],
        principal_df[pc_columns[2]],
        c=principal_df[pc_columns[3]],  # Use PC4 as the 4th dimension for shading
        cmap="viridis",  # Use a color spectrum colormap
        alpha=0.6,
    )
    ax.set_xlabel(pc_columns[0])
    ax.set_ylabel(pc_columns[1])
    ax.set_zlabel(pc_columns[2])
    ax.set_title(f"{title}")
    plt.colorbar(ax.scatter([], [], [], c=[], cmap="viridis"))  # Add a colorbar as a guide for the 4th dimension

    plt.tight_layout()
    fig.savefig(save_path + "scatter_plot.png")  # Save the PC4 shaded plot to a file
    plt.close(fig)

    # Scree Plot
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(range(1, n_components + 1), explained_variance_pc, marker="o")
    ax3.set_xlabel("Principal Components")
    ax3.set_ylabel("Explained Variance (%)")
    ax3.set_title("Scree Plot")
    plt.tight_layout(pad=2)
    fig3.savefig(save_path + "_scree.png")  # Save the scree plot to a file
    plt.close(fig3)

    # Feature Importance Plots for each PC
    for i in range(n_components):
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111)

        pc_feature_importance = feature_importance[i]
        sorted_indices = np.argsort(pc_feature_importance)[::-1]
        sorted_feature_names = [feature_names[j] for j in sorted_indices]
        sorted_feature_importance = pc_feature_importance[sorted_indices]

        ax2.barh(sorted_feature_names, sorted_feature_importance, color="#5ac8fa")  # Blue color for Apple
        ax2.set_xlabel("Feature Importance")
        ax2.set_ylabel("Features")
        ax2.set_title(f"PC{i+1} Feature Importance")

        plt.tight_layout(pad=3)
        fig2.savefig(save_path + f"_pc{i+1}_importance.png")  # Save the feature importance plot to a file
        plt.close(fig2)
