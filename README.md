# gaze_estimation_project
Final year M.Sc project

How to Run the applicaion?

1. install openCV,openvino
2. download blow modle from intel openvino public model
'face-detection-adas-0001'
'head-pose-estimation-adas-0001'
'gaze-estimation-adas-0002'
'facial-landmarks-35-adas-0002'
'age-gender-recognition-retail-0013'
3. change the model file path in gaze.py
4. change the arg 'CPU' in inference_network.load_model(modelfile+ '.xml', 'CPU', None) to 'GPU' or 'MYRIAD' if it applies.
5. offline mode change server_flag=False 
5.1 Online mode change server_flag=True, and do follwong steps
6. create a mqtt Broker on a server change the server ip, username passwrod  in mqtt_subscriber.py 
7. create database and change the corrsponding code in mqtt_subscriber.py and Web_plot.py (I am using PostgreSql here) 
