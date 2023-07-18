<h1 align="center"> YouTube Trending - MLOps </h1>
<!-- <h3 align="center"> An API that helps you indentify faces and emotions in images and videos </h3> -->

<div align="center">
  <img width="450" src="https://github.com/JenAlchimowicz/YouTube-Trending-MLops/assets/74935134/d3f7af57-4d79-448e-ac69-83a8acee93b5">
  <p align="center"><sub><a href="link/to/original/image">Image Source</a></sub></p>
</div>


<h2 id="project-description"> What is this repo about ❔ </h2>

`This repo shows how to deploy and manage machine learning models in production.`

Steps covered:
1. Define our problem and perform EDA
2. Develop an ETL pipeline
3. Train a model
4. Deploy the model to cloud
5. Develop and deploy a retraining pipeline
6. Monitor the model performance

The focus is on the `tools` and `ML best practices`. In particular, dockerizing and deploying to AWS the two key pipelines: retraining and inference. The problem itself - predicting YouTube views from just the channel name and video category - is rather trivial, and would usually be more complex in the real world. However, the methods of managing the ML lifecycle are relevant and can be used to deploy real-world projects.

Inference endpoint available at: [mlprojectsbyjen.com](https://mlprojectsbyjen.com)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="project-description"> :book: Table of contents </h2>

<ol>
  <li><a href="#repo-structure"> ➤ Repo structure </a></li>
  <li><a href="#inference"> ➤ Inference pipeline </a></li>
    <ul>
      <li><a href="#inference-infra"> AWS infrastructure </a></li>
      <li><a href="#inference-tools"> Tools </a></li>
    </ul>
  <li><a href="#retraining"> ➤ Retraining pipeline </a></li>
    <ul>
      <li><a href="#retraining-infra"> AWS infrastructure </a></li>
      <li><a href="#retraining-tools"> Tools </a></li>
    </ul>
</ol>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="repo-structure"> :pencil: Repo structure </h2>

The repo consists of 7 key components. Each is a small, separate project with it's own directory, requirements, docker image and AWS permissions.

- `configs` - congif files shared across all components  
- `data_ingestion` - loads data from the internet into S3  
- `data_transformation` - processes raw data. Outputs training datasets and artifacts  
- `dev` - directory with development files. Not used in the final product  
- `feature_store_update` - calculates features from raw data. Uploads features to DynamoDB  
- `predict_api` - host API server that outputs model predictions  
- `predict_monolith_deployment` - cost cutting measure. Merges `predict_api` and `web_endpoint` into one deployment  
- `training` - trains and tunes ML models  
- `web_endpoint` - user interface. Calls `prediction_api` for model predictions


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="inference"> :pencil: Inference pipeline </h2>

<h3 id="inference-infra"> AWS insfrastructure </h3>

The architecture follows a simple 2-tier design. The traffic flows from users to the external Application Load Balancer (ALB), which distributes it across Elastic Container Service (ECS) Tasks. When the user presses **predict** on the web app, a request is sent to the internal ALB. The App tier Tasks compute the ML predictions and return them back to the Web tier, where the results are displayed to the user.

</br>

<div align="center">
  <img width="700" src="https://github.com/JenAlchimowicz/YouTube-Trending-MLops/assets/74935134/be883aec-3e69-4fd3-91ec-91d49d6920a7">
</div>

</br>

\* Why is the App tier public? NAT Gateways are expensive for a small project such as this one - around 40$ per month per AZ. There are no major security concerns with this project and all components in the App tier have configured security groups to only allow traffic from relevant services.

** In reality there are 3 AZs configured

*** Depending on when you are reading this, the endpoind mlprojectsbyjen.com might actually use a monolith deployment on an EC2 instead of a 2-tier ECS-based architecture. It doesn't scale but it allows to avoid passive ALB charges, cutting down the costs from around $100 a month to around $10.


<h3 id="inference-tools"> Tools </h3>

The app itself uses standard ML python libraries: Pandas, scikit-learn, XGBoost, FastAPI and Streamlit. Neptune AI is used for experiment tracking and as a model registry. 

AWS Service choices:
- `Compute` - ECS for ease of deployment
- `Storage` - S3 for scalability and AWS integrations
- `Feature Store` - DynamoDB for quick read access
- `Scaling and High Availability` - ALB and ASG as they are the recommended standard in AWS
- `Access and security` - IAM Roles for AWS access and SSM Parameter Store for distributing keys for external services such as Neptune AI


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="retraining"> :pencil: Retraining pipeline </h2>

<h3 id="retraining-infra"> AWS insfrastructure </h3>
In progress...

<h3 id="retraining-tools"> Tools </h3>
In progress...
