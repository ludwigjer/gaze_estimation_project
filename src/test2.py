import psycopg2
import plotly
import plotly.graph_objects as go
import chart_studio.plotly as py
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html



DB_NAME='anihlpmu'
DB_USER ='anihlpmu'
DB_PASS ='8lCS6ZnrHsa6UE6DYEgi6gSukx8hnfwb'
DB_HOST ='kandula.db.elephantsql.com'
DB_PORT='5432'

try:
    conn = psycopg2.connect(database = DB_NAME, user = DB_USER,password = DB_PASS, host = DB_HOST, port =DB_PORT)
    print('Connected to PostgreSql')
except:
    print ('Unable to connect PostgreSql')

cur = conn.cursor()
cur.execute("SELECT * FROM GAZEDATA")
rows = cur.fetchall()

df = pd.DataFrame( [[ij for ij in i] for i in rows] )
df.rename(columns={0: 'id', 1: 'age', 2: 'gender', 3: 'timestamps', 4:'timelast'}, inplace=True);
df = df.sort_values(['timelast'], ascending=[1]);
id,age,gender,timestamps,timelast =df['id'],df['age'],df['gender'],df['timestamps'],df['timelast']

for i in range(len(rows)):
    print("timestamps : ", timestamps[i],", age : ", age[i],", gender : ", gender[i],", time_last : ", timelast[i])
print("The total number of people looked: " ,len(rows))


# Boostrap CSS.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

trace1 = go.Scatter( x=df['age'], y=df['timelast'], text=['gender'],mode='markers')
layout = go.Layout(
    title='Age vs Timelast',
    xaxis=dict( type='log', title='age' ),
    yaxis=dict( title='timelast' )
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)

fig_age_gender_histogram=px.histogram(df, x="age",title='Age Gender Histogram',color="gender",marginal="rug",hover_data=df.columns)
fig_timelast_gender_histogram=px.histogram(df, x="timelast", color="gender",marginal="rug",hover_data=df.columns)

def prepare_plot(query,type):
    print(type)
    cur.execute(query)
    rows = cur.fetchall()
    for i in rows:
        print("date: ", i[0], ", total: ", i[1])
    rows = np.array(rows)
    return rows

rows1 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  count(1) as people_per_day
from gazedata
group by 1 order by date_trunc
""",'Total per day')

rows2 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  count(1) as male_per_day
from gazedata where gender='Male'
group by 1 order by date_trunc
""",'Male per day')

rows3 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  count(1) as female_per_day
from gazedata where gender='Female'
group by 1 order by date_trunc
""",'Fale per day')
rows4 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  count(1) as people_per_day
from gazedata where age < 20
group by  date_trunc('day', timestamps) order by date_trunc;
""",'Age under 20 per day')
rows5 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  count(1) as people_per_day
from gazedata where age >= 20 and age <= 40 
group by  date_trunc('day', timestamps) order by date_trunc;
""",'Age 20 to 40 per day')
rows6 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  count(1) as people_per_day
from gazedata where age > 40
group by  date_trunc('day', timestamps) order by date_trunc;
""",'Age over 40 per day')
rows7 = prepare_plot("""
select
  date_trunc('day', timestamps), -- or hour, day, week, month, year
  sum(timelast) AS timelast
from gazedata
group by  date_trunc('day', timestamps) order by date_trunc;
""",'Total second per day')

line1 =go.Scatter(x=rows1[:,0],y=rows1[:,1], name="All people")
line2 =go.Scatter(x=rows2[:,0],y=rows2[:,1], name="Male")
line3 =go.Scatter(x=rows3[:,0],y=rows3[:,1], name="Female")
line4 =go.Scatter(x=rows4[:,0],y=rows4[:,1], name="Under 20")
line5 =go.Scatter(x=rows5[:,0],y=rows5[:,1], name="20 to 40")
line6 =go.Scatter(x=rows6[:,0],y=rows6[:,1], name="Over 40")
data_linegraph= [line1,line2,line3,line4,line5,line6]
layout = go.Layout(title='Number of People Among Different Group')
fig2=go.Figure(data=data_linegraph, layout=layout)



def generate_table(dataframe, max_rows=len(df)):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app.layout = html.Div(
                      children=[html.H1(
                          children='Gaze Information Analyse',
                          style={
                              'textAlign': 'center',
                              'color': colors['text']
                          }
                      ),
    html.Div(children='''
        Gaze Information Analyse Web Application.
    ''', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.Div(
            [
            html.Div([
                dcc.Graph(
                    id='graph1',
                    figure=fig)
                ], className= 'eight columns'
                ),
                html.Div([
                dcc.Graph(
                    id='graph3',
                    figure=fig_age_gender_histogram
                )
                ], className= 'four columns'
                )
            ], className="row"
        ),

    html.Div(
            [
            html.Div([
                dcc.Graph(
                    id='graph2',
                    figure=fig2 )
                ], className= 'eight columns'
                ),
                html.Div([
                dcc.Graph(
                    id='graph4',
                    figure=fig_timelast_gender_histogram
                )
                ], className= 'four columns'
                )
            ], className="row"
        ),
    generate_table(df)
], className='ten columns offset-by-one')


if __name__ == '__main__':
    app.run_server(debug=True)