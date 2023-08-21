# kickstarter-mlops

[<img src="images/localstack-banner.svg" width="150" height="28">](https://localstack.cloud/)
![Poetry](https://img.shields.io/badge/poetry-white?style=for-the-badge&logo=poetry&logoColor=%230daae2&labelColor=%2346528e&color=%2346528e)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![W&B](https://img.shields.io/badge/W%26B-black?style=for-the-badge&logo=weightsandbiases)
![Prefect](https://img.shields.io/badge/Prefect-white?style=for-the-badge&logo=Prefect&logoColor=blue)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 1. Context

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="images/kickstarter-logo.png" alt="Image Description" width="250" height="30">
  <div style="flex: 90%; padding: 10px;">
    <p>Kickstarter is a popular crowdfunding platform where creative minds 
    seek support for their projects. It hosts a wide variety of different 
    projects, spanning from technology startups to creative arts ventures 
    and social impact initiatives, among others. In Kickstarter, a project
    secures funds only if it reaches its predefined funding goal. Now, 
    there are numerous factors that influence the outcome of a project 
    (e.g., project category, funding goal, country, ...), making it feasible
    to develop a predictive model to determine a project's likelihood of success.</p>
  </div>
</div>



## 2. Goal

The objective of this project is to put into practice the knowledge acquired 
during the [mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) 
course (offered by [DataTalks.Club](https://datatalks.club/)) and constructing an MLOps pipeline to predict if a Kickstarter project
will succeed or fail.

## 3. Dataset

The latest available dataset is automatically downloaded from [webrobots.io](https://webrobots.io/kickstarter-datasets/).
It contains data on projects hosted on Kickstarter from 2009 to 2023. The raw data contains 39 features (see Jupyter Notebook), but 
we have selected 11 final features to feed our ML model (we think these are the most relevant):


| Feature                          | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| creation_to_launch_hours (float) | Time duration from project creation to launch on Kickstarter |
| campaign_hours (float)           | Time duration from launch to deadline                        |
| name_length (int)                | Length (in words) of the project's name                      |
| description_length (int)         | Length (in words) of the project's description               |
| usd_goal (float)                 | Funding goal in USD                                          |
| main_category (str)              | Project's main category (e.g., journalism)                   |
| sub_category (str)               | Project's sub-category (e.g., print)                         |
| country (str)                    | Acronym of country from which the project creator originates |
| staff_pick (bool)                | Indicates whether a project was highlighted as a staff pick  |
| diff_main_category_goal (float)  | Difference current goal wrt median goal of the main category |
| diff_sub_category_goal (float)   | Difference current goal wrt median goal of the sub-category  |

## 4. Prerequisites

<ol type="I">
  <li> <b><code>conda</code> (optional)</b> :   install <code>conda</code> on your system. </li>
  <li> <b><code>python</code></b> : make sure you have <code>python 3.10</code> installed (included in <code>conda</code>). </li>
  <li> <b><code>docker</code></b> : required for containerization. </li>
  <li> <b><code>docker compose</code></b> :  additionally, you'll need Compose V2 for managing multi-container applications. </li>
</ol> 





## 5. Setup

We will build everything on top of a baseline virtual env <code>(base)</code>. In our case it's provided by <code>conda</code>. 

### 5.1. Clone the repo
```bash
(base) $ git clone https://github.com/BoKatanKrize/kickstarter-mlops.git
```

### 5.2. Set up the Project using <code>poetry</code>. 
```bash
(base) $ pip install poetry
``` 
The idea is to use <code>poetry</code> as the package manager and dependency 
resolver while leveraging <code>conda</code> to manage the Python interpreter. Thus, if we run:
```bash
(base) $ poetry install
```
<code>poetry</code> has now created a virtual environment (on top of <code>(base)</code>) dedicated to the project, and 
installed the packages listed in <code>pyproject.toml</code>. The package dependencies are recorded in the lock file, 
named <code>poetry.lock</code>. 

Finally, install Poe the Poet as a poetry plugin 
```bash
(base) $ poetry self add 'poethepoet[poetry_plugin]'
```

### 5.3. Set up S3 bucket with <code>localstack</code>
1. Start <code>localstack</code> container 
```bash
(base) $ docker compose up -d localstack
```
2. Install the AWS Command Line Interface
```bash
(base) $ pip install awscli
```
3. Configure AWS CLI
   1. Make sure you have <code>localstack</code> up and running
   2. Run
   ```bash
    (base) $ aws configure --profile localstack-profile
   ```
   3. You'll be prompted to enter your AWS access key, secret key,
   region, and output format. E.g:
      - AWS Access Key ID [None]: localstack
      - Secret Access Key [None]: password
	  -	Default region name [None]: eu-west-1
      - Deault output format [None]: JSON
      
      which are the values saved in <code>sample.env</code>.
   4. The configuration will be stored in <code>~/.aws/config</code>
   5. Create <code>.env</code> based on <code>sample.env</code>  

### 5.4. Set up Weights & Biases
1. Create a W&B account
2. Create a project called <code>kickstarter-mlops</code>
3. Save the following variables to <code>.env</code>:
   1. Your W&B API Key
   2. Your entity (user name)
   
### 5.5. Set up Prefect Cloud
1. Create a prefect cloud account
2. Create an API key and a workspace
   1. Save them to <code>.env</code> 
   2. Finally, provide the [USER-ID] to complete <code>PREFECT_API_URL</code>
3. Authenticate 
```bash
    (base) $ prefect cloud login -k <your-api-key>
```

## 6. Training

One of the key advantages of using <code>poetry</code> is the streamlined management 
of script executions without the need for a <code>Makefile</code>. Poetry takes care 
of guiding the entire workflow from the <code>.git</code> root.

1. Launch <code>localstack</code> S3 bucket (if it's not running already)
```bash
(base) $ poetry poe launch-localstack-s3
```

2. The <code>downloader</code> script is responsible for downloading the latest data and saving it as raw data. It ensures that
you have the most up-to-date dataset to work with. When executed, it fetches the necessary data and stores it for 
further processing.
```bash
    (base) $ poetry run downloader
```
3. The <code>cleaner</code> script employs <code>scikit-learn</code> pipelines to clean the data and remove unnecessary features.
```bash
    (base) $ poetry run cleaner
```
4. The <code>build_features</code>  script utilizes <code>scikit-learn</code> pipelines to perform feature engineering. This step aims to enhance the quality of the final features
```bash
    (base) $ poetry run build_features
```
5. The <code>train</code> script employs Weights and Biases Sweep to perform hyperparameter optimization using both XGBoost and LightGBM. 
This step helps identify the best-performing model by tuning various hyperparameters.
```bash
    (base) $ poetry run train
```
6. The <code>register_model</code> script searches for the best model in terms of ROC AUC metric among XGBoost and LightGBM. Once identified, 
it stores this model in the W&B model registry, allowing for easy access and tracking.
```bash
    (base) $ poetry run register_model
```

Throughout each of these steps, the data and models are saved to an S3 bucket provided by Localstack, ensuring that all artifacts are 
preserved for future reference. E.g., to check the trained models from W&B Sweep:

```bash
    (base) $ aws s3 --endpoint-url http://localhost:4566 ls s3://kickstarter-bucket/models/trained/
```

### 7. Orchestration

The previous training workflow also can be automatically executed by using a Prefect deployment

1. Set up orchestration with Prefect
```bash
    (base) $ poetry poe setup-orchestration
```
   - Creates Prefect workpool
   - Creates Prefect deployment (unfortunately, can't skip the prompts; select default)
   - Starts Prefect worker
2. Launch Prefect orchestration
```bash
    (base) $ poetry poe setup-orchestration
```
   - It launches an <code>localstack</code> S3 bucket (if it's not running already)
   - It runs the Prefect deployment + workpool + worker
   - Best ML model is saved in W&B Registry 

### 8. Deployment as web-service

We deploy the best model in W&B Registry as a web-service (in a docker container). Make sure
that the S3 bucket from either direct training (sec. 6) or orchestration (sec. 7) is running. 
Then execute 
```bash
    (base) $ poetry poe launch-flask-app
```
to set up the web-service. The web service will automatically connect to the W&B Registry and get 
the best model. The web service works by reading a single Kickstarter project and 
its features (<code>sample_kickstarter_project.json</code>), and returning a prediction of 
"Successful" or "Failed". To send this packet of data and obtain the prediction, execute
```bash
    (base) $ poetry poe predict-flask-app
```


