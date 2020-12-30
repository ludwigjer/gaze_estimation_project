import sys
import math
import numpy as np
import cv2
import time
import subprocess
subprocess.call([r'C:\Program Files (x86)\Intel\openvino_2021.1.110\bin\setupvars.bat'])
import logging as log
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy import linalg as LA
from inference_network import Network
import paho.mqtt.client as paho


file_directory='E:\\Users\Administrator\\Documents\\PycharmProjects\\Gaze_Detection_Analyze_Application\\model\\'
file_det  = 'face-detection-0204'
file_rdf  = 'face-reidentification-retail-0095'
file_hp   = 'head-pose-estimation-adas-0001'
file_gaze = 'gaze-estimation-adas-0002'
file_lm   = 'facial-landmarks-35-adas-0002'
file_er = 'emotions-recognition-retail-0003'
file_ag = 'age-gender-recognition-retail-0013'
model_det  = file_directory+file_det
model_rdf  = file_directory+file_rdf
model_hp   = file_directory+file_hp
model_gaze = file_directory+file_gaze
model_lm   = file_directory+file_lm
model_er = file_directory+file_er
model_ag = file_directory+file_ag



def build_camera_matrix(center_of_face, focal_length):

    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1

    return camera_matrix

def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    #Degrees to radians
    yaw_radian = yaw * np.pi / 180.0
    pitch_radian =pitch * np.pi / 180.0
    roll_radian =roll* np.pi / 180.0

    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    Rx = np.array([[1,                0,                               0],
                   [0,                math.cos(pitch_radian),  -math.sin(pitch_radian)],
                   [0,                math.sin(pitch_radian),   math.cos(pitch_radian)]])
    Ry = np.array([[math.cos(yaw_radian),    0,                  -math.sin(yaw_radian)],
                   [0,                1,                               0],
                   [math.sin(yaw_radian),    0,                   math.cos(yaw_radian)]])
    Rz = np.array([[math.cos(roll_radian),   -math.sin(roll_radian),                 0],
                   [math.sin(roll_radian),   math.cos(roll_radian),                  0],
                   [0,                0,                               1]])

    #R = np.dot(Rz, np.dot(Ry, Rx))
    R = Rz @ Ry @ Rx
    #print(R)
    camera_matrix = build_camera_matrix(center_of_face, focal_length)

    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]

    xaxis = np.matmul(R, xaxis) + o
    yaxis = np.matmul(R, yaxis) + o
    zaxis = np.matmul(R, zaxis) + o
    zaxis1 = np.matmul(R, zaxis1) + o

    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    cv2.putText(frame, 'pitch'+str(pitch), (cx+100, cy), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    cv2.putText(frame, 'yaw' + str(yaw), (cx, cy-100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))

    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.putText(frame, 'roll' + str(roll), p2, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
    cv2.circle(frame, p2, 3, (255, 0, 0), 6)

    return frame


def preprocessing(input_frame, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    frame = cv2.resize(input_frame, (width, height))
    frame = frame.transpose((2,0,1))
    frame = frame.reshape(1, 3, height, width)

    return frame

def resize_fram(frame,ratio):
    scale = ratio / frame.shape[1]
    frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
    return frame

def draw_gaze_line(frame, coord1, coord2):
    cv2.line(frame, coord1, coord2, (0,0,255), 2)
    cv2.circle(frame, coord2, 10, (255, 255, 0), 15)

def get_detect_distance_ratio(width_panel,height_panel,face_percentage_of_frame):
    area=width_panel*height_panel
    ratio=face_percentage_of_frame/area
    return ratio

def most_frequent_gender(List):
    if len(List)>0:
        return max(set(List), key = List.count)

class Tracker:
    def __init__(self):
        # Identification information DB
        self.identifysDb = None
        # Face reliability DB
        self.conf = []

    def getIds(self, identifys, persons):
        if(identifys.size==0):
            return []
        if self.identifysDb is None:
            self.identifysDb = identifys
            for person in persons:
                self.conf.append(person[0])

        print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
        similaritys = self.__cos_similarity(identifys, self.identifysDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            persionId = ids[i]
            print("persionId:{} {} conf:{}".format(persionId,similarity[persionId],  persons[i][0]))
            #
            # If the reliability of face detection is 0.9 or higher and the reliability of face detection is higher
            # than the existing one, the identification information is updated.
            if(similarity[persionId] > 0.9 and persons[i][0] > self.conf[persionId]):
                print("? refresh id:{} conf:{}".format(persionId, persons[i][0]))
                self.identifysDb[persionId] = identifys[i]
            # If it is 0.3 or less, add it
            elif(similarity[persionId] < 0.3):
                self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
                self.conf.append(persons[i][0])
                ids[i] = len(self.identifysDb) - 1
                print("append id:{} similarity:{}".format(ids[i], similarity[persionId]))

        print(ids)
        # If there are duplicates, disable the less reliable one (unlikely)
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if(e == i):
                    continue
                if(a == b):
                    if(similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1
        print(ids)
        return ids


    def __cos_similarity(self, X, Y):
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (
            np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
        )




# Create a Network for using the Inference Engine
inference_network = Network()

# Prep for face detection
input_shape_det,out_shape_det=inference_network.load_model(model_det+ '.xml', 'CPU', None) # [1, 3, 384, 672] - [1,1,200,7]
input_name_det=inference_network.input_blob # Input blob name
out_name_det=inference_network.output_blob  # Output blob name
exec_net_det=inference_network.exec_network

# Prep for face reidentification
input_shape_rdf,out_shape_rdf=inference_network.load_model(model_rdf+ '.xml', 'CPU', None) # [1, 3, 128, 128] - [1,256,1,1]
input_name_rdf=inference_network.input_blob # Input blob name
out_name_rdf=inference_network.output_blob  # Output blob name
exec_net_rdf=inference_network.exec_network

# Preparation for headpose detection
input_shape_hp,out_shape_hp=inference_network.load_model(model_hp+ '.xml', 'CPU', None) # [1, 3, 60, 60] - [1, 1]
input_name_hp=inference_network.input_blob # Input blob name
out_name_hp=inference_network.output_blob  # Output blob name
exec_net_hp=inference_network.exec_network

# Preparation for landmark detection
input_shape_lm,out_shape_lm=inference_network.load_model(model_lm+ '.xml', 'CPU', None) # [1, 3, 62, 62] - [1, 1, 1, 1]
input_name_lm=inference_network.input_blob # Input blob name
out_name_lm=inference_network.output_blob  # Output blob name
exec_net_lm=inference_network.exec_network

# Preparation for Age/Gender Recognition
input_shape_ag,out_shape_ag=inference_network.load_model(model_ag+ '.xml', 'CPU', None) # [1,3,60,60] - [1, 70]
input_name_ag=inference_network.input_blob # Input blob name
out_name_ag=inference_network.output_blob  # Output blob name
exec_net_ag=inference_network.exec_network

# Preparation for gaze estimation
input_shape_gaze,out_shape_gaze=inference_network.load_model(model_gaze+ '.xml', 'CPU', None) # [1, 3, 60, 60] -
input_shape_gaze  = [1, 3, 60, 60]
exec_net_gaze = inference_network.exec_network

tracker = Tracker()




def main():

    scale = 150
    focal_length = 950.0
    height_panel=20
    width_panel=27
    mqtt_name = "ludwig"
    mqtt_pw = "000814"
    mqtt_ip ="192.168.0.88"
    port=1883
    client = paho.Client()
    client.username_pw_set(mqtt_name, mqtt_pw)
    gender_label = ('Female', 'Male')
    counter = 0
    counter_male=0
    counter_age_under20=0
    counter_age_20to40=0
    counter_age_over40=0
    gender_array = [[] for i in range(1000000)]
    age_array = [[] for i in range(1000000)]
    counter_array = [0]*100000
    time_start=[0]*100000
    time_end=[0]*100000
    time_last=[0]*100000
    total_time_array = [0]*100000
    server_flag=False
    flag=[False]*100000
    flag_male=[False]*100000
    flag_age_under20=[False]*100000
    flag_age_20to40=[False]*100000
    flag_age_over40=[False]*100000



    # Open USB webcams
    cam = cv2.VideoCapture(0)
    camx, camy = [(1920, 1080), (1280, 720), (800, 600), (480, 480)][1]  # Set camera resolution [1]=1280,720
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camx)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

    while True:
        START_TIME = time.time()
        gaze_lines = []
        ret, frame = cam.read()
        if ret == False:
            break
        if server_flag:
            client.connect(mqtt_ip, port, 60)

        scaler = max(int(frame.shape[0] / 1000), 1)
        local_time_array = []
        face_array=[]

        # frame = resize_fram(frame,640)
        out_frame = frame.copy()

        frame_h, frame_w = frame.shape[:2]
        # Face detection
        frame_preprocessed = preprocessing(out_frame,input_shape_det[2],input_shape_det[3])
        exec_net_det.infer(inputs={input_name_det: frame_preprocessed})

        # if exec_net_det.requests[0].wait(-1) == 0:
        res = exec_net_det.requests[0].outputs[out_name_det]

        # faces contains [image_id, label, conf, x_min, y_min, x_max, y_max]
        # [1,1,N,7] N= number of faces
        faces_frame = res[0][:, np.where(res[0][0][:, 2] > 0.8)]  # prob threshold : 0.5
        identifys = np.zeros((faces_frame.shape[2], 256))
        for i, face in enumerate(faces_frame[0][0]):
            box = [
                face[2],
                abs(int(face[3] * frame_w)),
                abs(int(face[4] * frame_h)),
                abs(int(face[5] * frame_w)),
                abs(int(face[6] * frame_h)),
            ]
            face_array.append(box)
            #face reidentification
            frame_preprocessed = preprocessing(frame[box[2]:box[4],box[1]:box[3]], input_shape_rdf[2], input_shape_rdf[3])
            exec_net_rdf.infer(inputs={input_name_rdf: frame_preprocessed})
            # (1, 256, 1, 1) => (256)
            result = (exec_net_rdf.requests[0].outputs[out_name_rdf])[0,:,0,0]
            # get similarity
            identifys[i] = result

        ids = tracker.getIds(identifys, face_array)

        # add box and indexes to face_frame
        for i, box in enumerate(face_array):
            if (ids[i] != -1):
                (xmin, ymin, xmax, ymax) = box[1:]
                face_frame = frame[ymin:ymax, xmin:xmax]
                area = ((ymax - ymin) * (xmax - xmin))
                face_percentage_of_frame = int(np.sqrt(area) / np.sqrt(frame_w * frame_h) * 100)
                detect_distance_ratio = get_detect_distance_ratio(width_panel, height_panel, face_percentage_of_frame)
                # if(detect_distance_ratio<0.035):
                #     print("error")
                #     continue
                cv2.rectangle(out_frame, (box[1], box[2]), (box[3], box[4]), (0, 255, 0), 2)
                cv2.putText(out_frame, "Face: " + str(ids[i]), (box[1], box[2] - 80), cv2.FONT_HERSHEY_SIMPLEX, .7,
                            (155, 0, 255), 2)
                print('Face index: ', ids[i],
                      'Face top-left corner point: ', xmin, ymin,
                      'Face bottom-right corner point: ', xmax, ymax,
                      'Percentage of the frame takes: ', face_percentage_of_frame, '%')

                # Head pose
                frame_preprocessed = preprocessing(face_frame, input_shape_hp[2], input_shape_hp[3])
                exec_net_hp.infer(inputs={input_name_hp: frame_preprocessed})
                # Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitсh or roll).
                yaw = exec_net_hp.requests[0].outputs['angle_y_fc'][0][0]
                pitch = exec_net_hp.requests[0].outputs['angle_p_fc'][0][0]
                roll = exec_net_hp.requests[0].outputs['angle_r_fc'][0][0]
                print("yaw:{:f}, pitch:{:f}, roll:{:f}".format(yaw, pitch, roll))
                center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
                draw_axes(out_frame, center_of_face, yaw, pitch, roll, scale, focal_length)


                #  Age/Gender Recognition
                frame_preprocessed = preprocessing(face_frame, input_shape_ag[2], input_shape_ag[3])
                exec_net_ag.infer(inputs={input_name_ag: frame_preprocessed})
                age = exec_net_ag.requests[0].outputs['age_conv3']
                prob = exec_net_ag.requests[0].outputs['prob']
                age = round(age[0][0][0][0] * 100,1)
                gender = gender_label[np.argmax(prob[0])]
                cv2.putText(out_frame, "Age: " + str(age), (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, .7,
                            (155, 200, 55), 2)
                cv2.putText(out_frame, "gender: " + gender, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (15, 200, 155), 2)


                # looking time
                if (-20 < yaw < 20) and (-30 < pitch < 15):
                    age_array[ids[i]].append(age)
                    gender_array[ids[i]].append(gender)
                    print("for ",ids[i], "age_array :" ,age_array[ids[i]])
                    time_start[ids[i]]=time.perf_counter()
                    local_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    local_time_array.append(local_time)
                    cv2.putText(out_frame, "you are looking" , (xmin, ymax+30), cv2.FONT_HERSHEY_SIMPLEX, .7, (155, 255, 255), 2)
                    cv2.putText(out_frame, "time last :"+str(abs(time_last[ids[i]])),(xmin, ymax + 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 12, 78), 2)
                    cv2.putText(out_frame, "your looked:" + str(counter_array[ids[i]])+" times",
                                (xmin, ymax + 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 54, 120), 2)
                    cv2.putText(out_frame, "total time :" + str(total_time_array[ids[i]])+" second",
                                (xmin, ymax + 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (45, 12, 166), 2)
                    if time_last[ids[i]]<-3:
                        flag[ids[i]]=True
                        if gender=='Male':
                            flag_male[ids[i]] = True
                        if age<20:
                            flag_age_under20[ids[i]] =True
                        if 20<=age<=40:
                            flag_age_20to40[ids[i]] =True
                        if age>40:
                            flag_age_over40[ids[i]] =True
                else:
                    time_end[ids[i]]=time.perf_counter()
                    cv2.putText(out_frame, "your looked:" + str(counter_array[ids[i]])+" times",
                                (xmin, ymax + 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 54, 120), 2)
                    cv2.putText(out_frame, "total time :" + str(total_time_array[ids[i]])+" second",
                                (xmin, ymax + 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (45, 12, 166), 2)

                    gender = most_frequent_gender(gender_array[ids[i]])
                    del age_array[ids[i]][:],gender_array[ids[i]][:]
                    if flag[ids[i]]:
                        counter+=1
                        counter_array[ids[i]]+=1
                        total_time_array[ids[i]]-=time_last[ids[i]]
                        print_msg='Log'+str(counter)+': '+\
                                 'Local time: '+ local_time+','+\
                                 ' gender: '+ gender+','+\
                                 ' age :'+ str(age)+','+ \
                                 ' Gaze last for '+ str(abs(time_last[ids[i]]))+ ' seconds.'
                        mqtt_msg=local_time+" "+str(counter)+" "+gender+" "+ str(age) +" "+ str(abs(time_last[ids[i]]))

                        print(print_msg)
                        try:
                            client.publish('info', mqtt_msg, 0)
                            client.disconnect()
                        except:
                            print("unable to connect the server")
                        flag[ids[i]]=False
                        if flag_male[ids[i]]:
                            counter_male+=1
                            flag_male[ids[i]]=False
                        if flag_age_under20[ids[i]]:
                            counter_age_under20+=1
                            flag_age_under20[ids[i]]=False
                        if flag_age_20to40[ids[i]]:
                            counter_age_20to40+=1
                            flag_age_20to40[ids[i]]=False
                        if flag_age_over40[ids[i]]:
                            counter_age_over40+=1
                            flag_age_over40[ids[i]]=False


                print('time starts looking:',time_start[ids[i]])
                print('time finishes looking:',time_end[ids[i]])
                time_last[ids[i]]=round(time_end[ids[i]] - time_start[ids[i]],2)
                if(time_last[ids[i]]<0):
                    print('looking last for : ', abs(time_last[ids[i]]), 'seconds')
                else:
                    print('not looking for : ', abs(time_last[ids[i]]), 'seconds')


                try:
                    male_percentage=(counter_male/counter) *100
                    female_percentage=100-male_percentage
                    age_under20_percentage=(counter_age_under20/counter) *100
                    age_20to40_percentage=(counter_age_20to40/counter) *100
                    age_over40_percentage=(counter_age_over40/counter) *100
                    #print(str(counter),"<20 :",str(counter_age_under20),"20-40  :",str(counter_age_20to40),"40 > :",str(counter_age_over40)," ",)
                except ZeroDivisionError:
                    male_percentage = 0
                    female_percentage=0
                    age_under20_percentage=0
                    age_20to40_percentage=0
                    age_over40_percentage=0

                cv2.putText(out_frame, 'Have been looked: ' +str(counter)+' times', (15 * scaler, 60 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 155, 255), 2)
                cv2.putText(out_frame, 'Male looked: ' + str(format(male_percentage, '.2f'))+ '%', (15 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 15, 15), 2)
                cv2.putText(out_frame, 'Female looked: ' + str(format(female_percentage, '.2f')) + '%',(15 * scaler, 140 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 155, 15), 2)
                cv2.putText(out_frame, 'Audience under 20: ' + str(format(age_under20_percentage, '.2f')) + '%',(15 * scaler, 180 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1, (123, 44, 15), 2)
                cv2.putText(out_frame, 'Audience 20 to 40: ' + str(format(age_20to40_percentage, '.2f')) + '%',(15 * scaler, 220 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1, (66, 244, 188), 2)
                cv2.putText(out_frame, 'Audience over 40: ' + str(format(age_over40_percentage, '.2f')) + '%',(15 * scaler, 260 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1, (111, 166, 233), 2)


        out_frame = np.clip(out_frame, 0, 255)
        out_frame = out_frame.astype('uint8')
        FPS = 1 / (time.time() - START_TIME)
        out_frame = cv2.putText(out_frame,
                            "FPS:{:.3f}".format(FPS),
                            (15 * scaler, 30* scaler), cv2.FONT_HERSHEY_SIMPLEX,
                            1* scaler, (255, 0, 0), 2 * scaler)

        cv2.imshow("gaze", out_frame)
        key = cv2.waitKey(1)
        if key == 27: break
    cv2.destroyAllWindows()

if __name__ == '__main__':
        sys.exit(main() or 0)