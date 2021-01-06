import paho.mqtt.client as paho
import psycopg2
import psycopg2.extras
import sys

DB_NAME='anihlpmu'
DB_USER ='anihlpmu'
DB_PASS ='8lCS6ZnrHsa6UE6DYEgi6gSukx8hnfwb'
DB_HOST ='kandula.db.elephantsql.com'
DB_PORT='5432'

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

    #add_to_PostgreSql(local_time, gender, age, time_last)

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




