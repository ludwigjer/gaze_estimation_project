import time
import paho.mqtt.client as paho
from firebase import firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import psycopg2
import psycopg2.extras




DB_NAME='anihlpmu'
DB_USER ='anihlpmu'
DB_PASS ='8lCS6ZnrHsa6UE6DYEgi6gSukx8hnfwb'
DB_HOST ='kandula.db.elephantsql.com'
DB_PORT='5432'

file_directory='C:\\Users\Administrator\\Documents\\Intel\\OpenVINO\\omz_demos_build\\intel64\\Release\\gaze_estimation_Project\\'
#url='https://openvino-porject-default-rtdb.firebaseio.com/'
key='openvino-porject-firebase-adminsdk-m04su-49cb856165.json'
#firebase =firebase.FirebaseApplication(url)
cred = credentials.Certificate(file_directory+key)
firebase_admin.initialize_app(cred)
db = firestore.client()

#PostgreSql
try:
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    print('Connected to PostgreSql')
except:
    print ('Unable to connect PostgreSql')
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# Callback Function on Receiving the Subscribed Topic/Message
def on_message(client, userdata, msg):
    # print the message received from the subscribed topic
    message = msg.payload.decode()
    message_array=message.split()
    counter=message_array[2]
    local_time=message_array[0]+" "+message_array[1]
    gender=message_array[3]
    age=message_array[4]
    time_last=message_array[5]
    print('topic :'+ msg.topic + ", " + \
          'Log' + counter + ': ' + \
               'Local time: ' + local_time + ',' + \
               ' gender: ' + gender + ',' + \
               ' age :' + age + ',' + \
               ' Gaze last for ' + time_last + ' seconds.')

    # firebase.put("/"+counter, "timestamps", local_time)
    # add_to_firebase(local_time, gender, age, time_last)
    add_to_PostgreSql(local_time, gender, age, time_last)

def add_to_firebase(local_time,gender,age,time_last):
    doc_ref = db.collection('gaze_data').document(local_time)
    doc_ref.set({
        'timestamps':local_time,
        'gender': gender,
        'age': float(age),
        'gaze_last':  float(time_last)
    })

def add_to_PostgreSql(local_time,gender,age,time_last):
    # PostgreSql
    try:
        conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        print('Connected to PostgreSql')
    except:
        print ('Unable to connect PostgreSql')
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("insert into GAZEDATA (age, gender, timestamps, timelast) values ("+str(age)+",'"+gender+"','"+local_time+"',"+str(time_last)+")")
    conn.commit()
    print('Data inserted successfully')
    conn.close()

# stream Firebase
# emp_ref = db.collection('gaze_data')
# docs = emp_ref.stream()
# for doc in docs:
#     print('{} => {}'.format(doc.id,doc.to_dict()))

#
cur.execute("SELECT * FROM GAZEDATA")
rows = cur.fetchall()
for data in rows:
    print("timestamps : ", data[3],", age : ",data[1],", gender : ", data[2],", time_last : ", data[4])
print("The total number of people looked: " ,len(rows))







#MQTT
client = paho.Client()
client.on_message = on_message

client.username_pw_set("ludwig", "000814")
if client.connect("192.168.0.88", 1883, 60) !=0:
    print("error")
    sys.exit(-1)
client.subscribe("info")
try:
    print("CTRL + C to exit")
    client.loop_forever()
except:
    print("Disconnecting form broker")
client.disconnect()




