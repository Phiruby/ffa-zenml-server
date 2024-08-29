
zenml service-connector register aws_connector \
 --type aws \
 --auth-method iam-role \
 --role_arn=${ZENML_ROLE_ARN} \
 --region=${REGION} \
 --aws_access_key_id=${ZENML_USER_ACCESS_KEY_ID} \
 --aws_secret_access_key=${ZENML_USER_SECRET_KEY}

zenml artifact-store register s3_artifact_store -f s3 --path=${ZENML_ARTIFACT_STORE_BUCKET} --connector aws_connector   

# zenml orchestrator register sagemaker-orchestrator --flavor=sagemaker --region=${REGION} --execution_role=${ZENML_ROLE_ARN}   
# zenml orchestrator register sagemaker-orchestrator --flavor=sagemaker --aws_region=${REGION} --execution_role=${ZENML_ROLE_ARN}   
zenml orchestrator register sagemaker-orchestrator \
    --flavor=sagemaker \
    --execution_role=${ZENML_ROLE_ARN} \
    --aws_access_key_id=${ZENML_USER_ACCESS_KEY_ID} \
    --aws_secret_access_key=${ZENML_USER_SECRET_KEY} \
    --aws_region=${REGION}


zenml container-registry register ecr-registry --flavor=aws --uri=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com --connector aws_connector

zenml experiment-tracker register mlflow_ec2_tracker --flavor=mlflow --tracking_uri=${MLFLOW_TRACKING_URI} \
 --tracking_username="e" \
 --tracking_password="e"

zenml model-registry register mlflow_model_registry --flavor=mlflow

zenml stack register aws_stack -o sagemaker-orchestrator \
    -a s3_artifact_store -c ecr-registry -e mlflow_ec2_tracker -r mlflow_model_registry --set

zenml orchestrator update sagemaker-orchestrator --synchronous=False # for long running pipelines (eg: training)

echo "Successfully created stack"