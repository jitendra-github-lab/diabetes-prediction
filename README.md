# diabetes-prediction
To run diabetes-prediction please follow the below steps (commands) - 
$ git clone https://github.com/jitendra-github-lab/diabetes-prediction.git
$ cd diabetes-prediction/kubedemo/app
$ docker build -f Dockerfile -t diabetes-score:latest .
$ kubectl apply -f deployment.yaml
$ kubectl get all

Note: From service take the newly generated ip:port and fire on browser
