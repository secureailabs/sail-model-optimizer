import io
import os
import string

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sail_model_optimizer.visualizations.visualizations import (
    calculate_results,
    generate_feature_wordcloud,
    plot_feature_venn,
    plot_roc_curves,
    train_model_from_file,
    visualize_pca,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/roc_curves")
def validation(params: dict):

    os.environ["PATH_DIR_DATA"] = params["PATH_DIR_DATA"]
    os.environ["PATH_DIR_RESULTS"] = params["PATH_DIR_RESULTS"]
    PATH_DIR_DATA = os.environ["PATH_DIR_DATA"]
    PATH_DIR_RESULTS = os.environ["PATH_DIR_RESULTS"]
    model_names = params["model_names"]
    print(model_names)

    df_input_train = pd.read_csv(PATH_DIR_DATA + "input_train.csv")
    df_input_test = pd.read_csv(PATH_DIR_DATA + "input_test.csv")
    df_output_train = pd.read_csv(PATH_DIR_DATA + "output_train.csv")
    df_output_test = pd.read_csv(PATH_DIR_DATA + "output_test.csv")

    # Load previous models
    model_data = []
    feature_lists = []
    for name in model_names:
        path_file_results = os.path.join(PATH_DIR_RESULTS, name + ".json")

        model = None
        print(name)
        if name == "xgboost":
            model = xgb.XGBClassifier
        elif name == "logistic":
            model = LogisticRegression
        elif name == "bayesian":
            model = GaussianNB
        else:
            return {"Error": "Model," + name + ", name not found"}

        model, features = train_model_from_file(path_file_results, model, df_input_train, df_output_train)
        # feature_lists.append(features)

        array_input_test = df_input_test[features].to_numpy()
        model_data.append((model, array_input_test, name))

    results = calculate_results(model_data, df_output_test)
    fig = plot_roc_curves(results)

    # feature_lists.append(features)
    # set_labels = (model_names)
    # plot_feature_venn(feature_lists, set_labels)

    # create a buffer to store image data
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/venn_diagram")
def validation(params: dict):

    os.environ["PATH_DIR_DATA"] = params["PATH_DIR_DATA"]
    os.environ["PATH_DIR_RESULTS"] = params["PATH_DIR_RESULTS"]
    PATH_DIR_DATA = os.environ["PATH_DIR_DATA"]
    PATH_DIR_RESULTS = os.environ["PATH_DIR_RESULTS"]
    model_names = params["model_names"]
    print(model_names)

    df_input_train = pd.read_csv(PATH_DIR_DATA + "input_train.csv")
    df_input_test = pd.read_csv(PATH_DIR_DATA + "input_test.csv")
    df_output_train = pd.read_csv(PATH_DIR_DATA + "output_train.csv")
    df_output_test = pd.read_csv(PATH_DIR_DATA + "output_test.csv")

    # Load previous models
    model_data = []
    feature_lists = []
    for name in model_names:
        path_file_results = os.path.join(PATH_DIR_RESULTS, name + ".json")
        model = None

        print(name)
        if name == "xgboost":
            model = xgb.XGBClassifier
        elif name == "logistic":
            model = LogisticRegression
        elif name == "bayesian":
            model = GaussianNB
        else:
            return {"Error": "Model," + name + ", name not found"}

        model, features = train_model_from_file(path_file_results, model, df_input_train, df_output_train)
        feature_lists.append(features)
        array_input_test = df_input_test[features].to_numpy()
        model_data.append((model, array_input_test, name))

    results = calculate_results(model_data, df_output_test)
    # fig = plot_roc_curves(results)

    # feature_lists.append(features)
    set_labels = model_names
    plt = plot_feature_venn(feature_lists, set_labels)

    # create a buffer to store image data
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
