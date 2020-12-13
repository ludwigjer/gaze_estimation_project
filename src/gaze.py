import sys
import math
import random
import numpy as np
from PIL import Image
from numpy import linalg as LA
import cv2
import os
import logging as log
import time
from scipy.spatial import distance
from openvino.inference_engine import IECore
import matplotlib.pyplot as plt
import paho.mqtt.client as paho
from firebase import firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

#test
file_directory='C:\\Users\Administrator\\Documents\\Intel\\OpenVINO\\omz_demos_build\\intel64\\Release\\gaze_estimation_Project\\'
file_det  = 'face-detection-adas-0001'
file_hp   = 'head-pose-estimation-adas-0001'
file_gaze = 'gaze-estimation-adas-0002'
file_lm   = 'facial-landmarks-35-adas-0002'
file_er = 'emotions-recognition-retail-0003'
file_ag = 'age-gender-recognition-retail-0013'
model_det  = file_directory+file_det
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
    print(R)
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

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        # network = IENetwork(model=model_xml, weights=model_bin)
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))

        # Get the output layer
        self.output_blob = next(iter(self.network.outputs))

        # Return the input and output shape (to determine preprocessing)
        return self.network.inputs[self.input_blob].shape,self.network.outputs[self.output_blob].shape

    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=0,
            inputs={self.input_blob: image})
        return

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status


    def sync_inference(self, image):
        '''
        Makes a synchronous inference request, given an input image.
        '''
        self.exec_network.infer({self.input_blob: image})
        return


    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs


# Create a Network for using the Inference Engine
inference_network = Network()

# Prep for face detection
input_shape_det,out_shape_det=inference_network.load_model(model_det+ '.xml', 'CPU', None) # [1, 3, 384, 672] - [1,1,200,7]
input_name_det=inference_network.input_blob # Input blob name
out_name_det=inference_network.output_blob  # Output blob name
exec_net_det=inference_network.exec_network

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
    time_start=[0]*100
    time_end=[0]*100
    time_last=[0]*100
    server_flag=True
    flag=[False]*100
    flag_male=[False]*100
    flag_age_under20=[False]*100
    flag_age_20to40=[False]*100
    flag_age_over40=[False]*100

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
        time_start_array = []
        time_end_array = []
        local_time_array = []
        time_last_array = []
        flag_array =[]

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
        faces = res[0][:, np.where(res[0][0][:, 2] > 0.75)]  # prob threshold : 0.5


        #  find faces
        for i, face in enumerate(faces[0][0]):
            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            (xmin, ymin, xmax, ymax) = abs(box.astype("int"))
            area = ((ymax - ymin) * (xmax - xmin))
            face_percentage_of_frame=int(np.sqrt(area) / np.sqrt(frame_w * frame_h) * 100)
            detect_distance_ratio = get_detect_distance_ratio(width_panel, height_panel,face_percentage_of_frame)
            if(detect_distance_ratio<0.0):
                continue
            print('Face index: ',i,
                  'Face top-left corner point: ', xmin,ymin,
                  'Face bottom-right corner point: ',xmax, ymax,
                  'Percentage of the frame takes: ',face_percentage_of_frame,'%')
            # store face area
            face_frame = frame[ymin:ymax, xmin:xmax]
            # draw faces
            cv2.rectangle(out_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(out_frame, "Face: "+ str(i), (xmin, ymin-80), cv2.FONT_HERSHEY_SIMPLEX, .7, (155, 0, 255), 2)



            # Head pose
            frame_preprocessed = preprocessing(face_frame, input_shape_hp[2], input_shape_hp[3])
            exec_net_hp.infer(inputs={input_name_hp: frame_preprocessed})

            yaw = .0  # Axis of rotation: y
            pitch = .0  # Axis of rotation: x
            roll = .0  # Axis of rotation: z
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
            cv2.putText(out_frame, "gender: " + gender, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (15, 200, 155),
                        2)

            # looking time
            if (-20 < yaw < 20) and (-30 < pitch < 15):
                time_end_array.append(time_end[i])
                time_start[i]=time.time()
                time_start_array.append(time_start[i])
                local_time = time.strftime('%Y-%m-%d %H:%M:%S')
                local_time_array.append(local_time)
                print(gender)
                cv2.putText(out_frame, "you are looking" , (xmin, ymax+30), cv2.FONT_HERSHEY_SIMPLEX, .7, (155, 255, 255), 2)
                if time_last[i]<-3:
                    flag[i]=True
                    if gender=='Male':
                        flag_male[i] = True
                    if age<20:
                        flag_age_under20[i] =True
                    if 20<=age<=40:
                        flag_age_20to40[i] =True
                    if age>40:
                        flag_age_over40[i] =True
            else:
                time_start_array.append(time_start[i])
                time_end[i]=time.time()
                time_end_array.append(time_end[i])
                if flag[i]:
                    counter+=1

                    print_msg='Log'+str(counter)+': '+\
                             'Local time: '+ local_time+','+\
                             ' gender: '+ gender+','+\
                             ' age :'+ str(age)+','+ \
                             ' Gaze last for '+ str(abs(time_last[i]))+ ' seconds.'
                    mqtt_msg=local_time+" "+str(counter)+" "+gender+" "+ str(age) +" "+ str(abs(time_last[i]))

                    print(print_msg)
                    try:
                        client.publish('info', mqtt_msg, 0)
                        client.disconnect()
                    except:
                        print("unable to connect the server")

                    flag[i]=False
                    if flag_male[i]:
                        counter_male+=1
                        flag_male[i]=False
                    if flag_age_under20[i]:
                        counter_age_under20+=1
                        flag_age_under20[i]=False
                    if flag_age_20to40[i]:
                        counter_age_20to40+=1
                        flag_age_20to40[i]=False
                    if flag_age_over40[i]:
                        counter_age_over40+=1
                        flag_age_over40[i]=False

            print('time starts looking:',time_start_array[i])
            print('time finishes looking:',time_end_array[i])
            time_last[i]=round(time_end[i] - time_start[i],2)
            time_last_array.append(time_last[i])
            if(time_last_array[i]<0):
                print('looking last for : ', abs(time_last_array[i]), 'seconds')
            else:
                print('not looking for : ', abs(time_last_array[i]), 'seconds')

            try:
                male_percentage=(counter_male/counter) *100
                female_percentage=100-male_percentage
                age_under20_percentage=(counter_age_under20/counter) *100
                age_20to40_percentage=(counter_age_20to40/counter) *100
                age_over40_percentage=(counter_age_over40/counter) *100
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






        # Use Landmark to find eyes
            frame_preprocessed = preprocessing(face_frame, input_shape_lm[2], input_shape_lm[3])

            exec_net_lm.infer(inputs={input_name_lm: frame_preprocessed})

            normed_landmarks = (exec_net_lm.requests[0].outputs[out_name_lm][0])[0:8]
            eye_landmarks = normed_landmarks.reshape(4, 2)

            eye_x = 0
            eye_y = 1
            # Landmark position memo...
            eye_sizes = [abs(int((eye_landmarks[0][eye_x] - eye_landmarks[1][eye_x]) * face_frame.shape[1])),
                         abs(int((eye_landmarks[3][eye_x] - eye_landmarks[2][eye_x]) * face_frame.shape[
                             1]))]  # eye size in the cropped face image
            eye_centers = [[int(((eye_landmarks[0][eye_x] + eye_landmarks[1][eye_x]) / 2 * face_frame.shape[1])),
                            int(((eye_landmarks[0][eye_y] + eye_landmarks[1][eye_y]) / 2 * face_frame.shape[0]))],
                           [int(((eye_landmarks[3][eye_x] + eye_landmarks[2][eye_x]) / 2 * face_frame.shape[1])),
                            int(((eye_landmarks[3][eye_y] + eye_landmarks[2][eye_y]) / 2 * face_frame.shape[
                                0]))]]  # eye center coordinate in the cropped face image

            if eye_sizes[0] < 4 or eye_sizes[1] < 4:
                continue
            eyebox_size_ratio = 0.7
            eyes = []
            for i in range(2):
                # Crop eye images
                x1 = int(eye_centers[i][eye_x] - eye_sizes[i] * eyebox_size_ratio)
                x2 = int(eye_centers[i][eye_x] + eye_sizes[i] * eyebox_size_ratio)
                y1 = int(eye_centers[i][eye_y] - eye_sizes[i] * eyebox_size_ratio)
                y2 = int(eye_centers[i][eye_y] + eye_sizes[i] * eyebox_size_ratio)
                # crop and resize
                eyes.append(cv2.resize(face_frame[y1:y2, x1:x2].copy(),
                                           (input_shape_gaze[3], input_shape_gaze[2])))
                # Draw eye boundary boxes
                cv2.rectangle(out_frame, (x1 + xmin, y1 + ymin), (x2 + xmin, y2 + ymin), (0, 255, 0), 2)

                # rotate eyes around Z axis to keep them level
                if roll != 0.:
                    rotMat = cv2.getRotationMatrix2D((int(input_shape_gaze[3] / 2), int(input_shape_gaze[2] / 2)), roll, 1.0)
                    eyes[i] = cv2.warpAffine(eyes[i], rotMat, (input_shape_gaze[3], input_shape_gaze[2]),flags=cv2.INTER_LINEAR)

                # Change data layout from HWC to CHW
                eyes[i] = eyes[i].transpose((2, 0, 1))
                eyes[i] = eyes[i].reshape((1, 3, 60, 60))
            # head pose angle in degree
            hp_angle = [yaw, pitch, roll]

            # gaze estimation
            res_gaze = exec_net_gaze.infer(inputs={'left_eye_image': eyes[0],
                                                   'right_eye_image': eyes[1],
                                                   'head_pose_angles': hp_angle})
            # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
            gaze_vec = res_gaze['gaze_vector'][0]
            # normalize the gaze vector
            gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)
            vcos = math.cos(math.radians(roll))
            vsin = math.sin(math.radians(roll))
            tmpx = gaze_vec_norm[0] * vcos + gaze_vec_norm[1] * vsin
            tmpy = -gaze_vec_norm[0] * vsin + gaze_vec_norm[1] * vcos
            gaze_vec_norm = [tmpx, tmpy]
            # Store gaze line coordinations
            for i in range(2):
                coord1 = (eye_centers[i][eye_x] + xmin, eye_centers[i][eye_y] + ymin)
                coord2 = (eye_centers[i][eye_x] + xmin + int((gaze_vec_norm[0] + 0.) * 3000),
                          eye_centers[i][eye_y] + ymin - int((gaze_vec_norm[1] + 0.) * 3000))
                gaze_lines.append([coord1, coord2])

            # Drawing gaze lines
            for gaze_line in gaze_lines:
                draw_gaze_line(out_frame, (gaze_line[0][0], gaze_line[0][1]), (gaze_line[1][0], gaze_line[1][1]))


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