# ffa-zenml-server
1. ```pip install zenml```
2. ```pip install "zenml[server]"```
3. ```zenml integration install mlflow -y```

(Note that zenml's MLFlow model registry will use the same configurations as the MLFlow experiment tracker)

Let's discuss how to setup with an AWS stack. TO do so, refer the following documentation: https://docs.zenml.io/how-to/popular-integrations/aws-guide (make sure to run ```pip install zenml[aws]``` before installing zenml integration with s3)
You may also need to refer to the following before installing zenml s3: https://docs.zenml.io/how-to/auth-management/aws-service-connector

Now, to setup the MLFlow server, you can go through the following (to be configured with the cloud): https://mlflow.org/docs/latest/tracking/server.html

Once the MLFlow server is set up, read the following to configure with ZenML: https://docs.zenml.io/stack-components/experiment-trackers/mlflow#authentication-methods

