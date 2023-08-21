import os
import wandb
import json
import pickle
import pandas as pd

from flask import Flask, jsonify, request


AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_INTERIM_MODELS = os.getenv("WANDB_INTERIM_MODELS")
WANDB_PROCESSED_MODELS = os.getenv("WANDB_PROCESSED_MODELS")
WANDB_REGISTERED_MODELS = os.getenv("WANDB_REGISTERED_MODELS")


app = Flask(WANDB_PROJECT)


# def load_preprocessing_pipeline(api, option):
#     """
#     Loads a preprocessing pipeline from W&B
#     """
#
#     model_artifact = api.artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{option}:v0', type='model')
#     os.makedirs(f'{option}', exist_ok=True)
#     path = f"./{option}"
#     model_artifact.download(root=path)
#     with open(f'{path}/model.pkl', 'rb') as file:
#         pipe = pickle.load(file)
#     return pipe


def load_model_from_registry(api):
    """
    Loads the ML model from the W&B registry
    """
    model_artifact = api.artifact(f'{WANDB_ENTITY}/model-registry/{WANDB_REGISTERED_MODELS}:v0', type='model')
    model_artifact.download(root="./")
    with open(f'{WANDB_REGISTERED_MODELS}.pkl', 'rb') as file:
        registered_model = pickle.load(file)
    return registered_model


#def prepare_data(records_json, cleaner_pipe, engineer_pipe):
def prepare_data(records_json):
    # Parse the JSON data into a Python dictionary
    data_dict = json.loads(records_json)
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    # Cleaning records
    # df = cleaner_pipe.transform(df)
    # # Feature engineering the records
    # df = engineer_pipe.transform(df)
    # Extract the target column and divide X,y
    # target = 'state'
    # y = df[target]
    # X = df.drop(target, axis=1)
    return df


def predict(model, preprocessed_data):

    prediction = model.predict(preprocessed_data)

    if prediction[0] == 1:
        return "Successful"
    else:  # == 0
        return "Failed"


@app.route("/predict", methods=['POST'])
def predict_endpoint():

    wandb.login(key=WANDB_API_KEY)
    api = wandb.Api()

    records = request.get_json()
    print(records)
    # Loads the cleaning and feat. engineering pipelines from W&B
    # cleaner_pipe = load_preprocessing_pipeline(api, WANDB_INTERIM_MODELS)
    # engineer_pipe = load_preprocessing_pipeline(api, WANDB_PROCESSED_MODELS)

    # Clean and feature engineer the records
    # X, y = prepare_data(records, cleaner_pipe, engineer_pipe)
    X = prepare_data(records)

    # Loads model from Model Registry
    registered_model = load_model_from_registry(api)

    # Predict the outcome of Kickstarter project
    outcome = predict(registered_model, X)

    return jsonify({"Kickstarter Project": outcome})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9696)
