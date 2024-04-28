AWS_ACCOUNT_ID=052567997892
#aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com
#docker build --push . -t $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/llama-factory:latest
#docker build . -t $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/llama-factory:latest

AWS_ACCOUNT_ID=052567997892
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com
docker build . -t $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker:zw
docker push $AWS_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/ecr-aiss-sagemaker:zw