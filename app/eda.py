#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:39:43 2022

@author: arshdeepsingh
"""

import streamlit as st
import numpy as np
import pandas as pd
# from data.create_data import create_table

#add an import to Hydralit
from hydralit import HydraHeadApp


import streamlit as st

import datetime
from streamlit_folium import folium_static
import folium
import numpy as npfolium
import folium
import config
import random
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from datetime import datetime, timedelta
from plotly.graph_objs import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from st_aggrid.shared import JsCode
import numpy as np
from datetime import datetime
import torch.nn as nn
import os
import config
import torch
import h5py
from model import HotspotPredictor


    
###LOAD ML MODEL
def load_model(model_path, model):
    trained = torch.load(model_path,map_location=torch.device('cpu') )
    model.load_state_dict(trained['model'])
    print('\n Model Loaded \n')

##READ FEAR+TURES AND TARGETS
def read_h5(data_path, name):
        '''
        Read a h5 file

        Inputs: data_path <str> : path to data
                name <str> : dataset name in h5 file
        Output: arr <np.array> : numpy array
        '''
        hf = h5py.File(data_path, 'r')
        arr = np.array(hf[name][:])
        return arr

### Generate LATLING BINS
def getBins(min_,max_,n_bins):
    bins = np.linspace(start=min_, stop=max_, num=n_bins+1)
    return bins


#create a wrapper class
class EDA(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
       st.markdown("<h2 style='text-align: Left; color: black;'>Crime and Extraneous Factors Trends</h2>", unsafe_allow_html=True)
       # st.title("Historic Data")
       startdate=datetime(2020, 1, 17)
       enddate=datetime(2022,2,9)
       
       st.sidebar.markdown(f'<div style="color:black;text-align: center;font-size:16px;"><b>{"Compare trends in crime with other factors? "}</b></div>', unsafe_allow_html=True)
       desc='<br> Choose a date and historical time span to check trends using the fields below. <br> The following external factors are compared with crime: <ul> <li style="font-size:12px">Max and Min Temperature</li>  <li style="font-size:12px">Housing Price Index (HPI)</li> <li style="font-size:12px">Consumer Price Index (CPI)</li> <li style="font-size:12px">Max and Min Snow</li> </ul>'

       st.sidebar.markdown(f'<div style="color:black;text-align: left;font-size:13px;justify-content:justify">{desc}</div>', unsafe_allow_html=True)




       date_eda = st.sidebar.date_input("Choose a date",  min_value=startdate, max_value=enddate,value=startdate)


       values=['3 Months', '6 Months', '9 Months','12 Months','18 Months','24 Months']
       default_ix = values.index('12 Months')

       selection_box = st.sidebar.selectbox( "Select duration ", values,index=default_ix)
       with st.sidebar:
           cola1,cola2,cola3=st.sidebar.columns([1,5,1])
           cola1,cola2,cola3=st.columns(3)
           
           
           with cola2:
               
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               # st.write("")
              
        
           
               image = Image.open('logo.svg.png')
               cola2.image(image, caption='')

       vpd='Vancouver Police Department'
       st.sidebar.markdown(f'<div style="color:black;text-align: center;font-size:12px;justify-content:justify">{vpd}</div>', unsafe_allow_html=True)
               

       # image = Image.open('logo.svg.png')
       # st.sidebar.image(image, caption='',width=150,use_column_width='never')
       
       
       col1, col2,col3 = st.columns([5,0.5,5])
       data=pd.read_csv('EDA_Input.csv')
       date_max = datetime.strptime(str(date_eda),'%Y-%m-%d')
       #input from dropdown
       
       
       m=int(selection_box[0:len(selection_box)-7])
       d=m*31
       date_min = date_max - timedelta(d)
       
       
       
       data['mydate']=pd.to_datetime(data['monthyear'])
       
       data=data[(data['mydate']>= date_min) & (data['mydate']<=date_max)]
       # if date_min>startdate:
       #     data=data[(data['mydate']>= date_min) & (data['mydate']<=date_max)]
       # else:
       #     data=data[(data['mydate']>= startdate) & (data['mydate']<=date_max)]

       
      
       with col1:
           
           
           ########1
           c=st.container()
           fig = make_subplots(specs=[[{"secondary_y": True}]])
            
        # Add traces
           fig.add_trace(
            go.Scatter(x=data["monthyear"], y=data["crimecount"], name="# Crime Incidents"),
            secondary_y=False
        )
        
           fig.add_trace(
            go.Scatter(x=data["monthyear"], y=data["min_temperature"], name="Min. Temp. (°C)",line_color='green'),
            secondary_y=True
        )
        
        # Add figure title
           fig.update_layout(
            title_text="<b>Crime </b>and <b> Minimum Temperature </b>"
        )
        
        # Set x-axis title
           fig.update_xaxes(title_text="Year")
        
        # Set y-axes titles
           fig.update_yaxes(title_text="<b># Crime Incidents</b>", secondary_y=False)
           fig.update_yaxes(title_text="<b>Min. Temp. (°C)</b>", secondary_y=True)
           fig.update_layout(xaxis=dict(showgrid=False),
           yaxis=dict(showgrid=False),paper_bgcolor="white",width=int(550),
           legend=dict(font = dict(size = 8),yanchor="top", y=1.2, xanchor="left", x=.7))
        
           fig['layout']['yaxis2']['showgrid'] = False
           c.plotly_chart(fig) 
       
           ### graph 1
           fig = make_subplots(specs=[[{"secondary_y": True}]])

           # Add traces
           fig.add_trace(
               go.Scatter(x=data["monthyear"], y=data["crimecount"], name="# Crime Incidents"),
               secondary_y=False
           )
           
           fig.add_trace(
               go.Scatter(x=data["monthyear"], y=data["housing price indexes"], name="HPI",line_color='red'),
               secondary_y=True
           )
           
           # Add figure title
           fig.update_layout(
               title_text="<b>Crime </b> and <b> Housing Price Index (HPI) </b>"
           )
           
           # Set x-axis title
           fig.update_xaxes(title_text="Year")
           
           # Set y-axes titles
           fig.update_yaxes(title_text="<b># Crime Incidents</b>", secondary_y=False)
           fig.update_yaxes(title_text="<b>HPI</b>", secondary_y=True)
           fig.update_layout(xaxis=dict(showgrid=False),
                         yaxis=dict(showgrid=False),paper_bgcolor="white",
                         width=int(550),
                         legend=dict(font = dict(size = 8),yanchor="top", y=1.2, xanchor="left", x=.7))
           
           fig['layout']['yaxis2']['showgrid'] = False
           c.plotly_chart(fig)
           
           ###Graph 3
           fig = make_subplots(specs=[[{"secondary_y": True}]])
           
           # Add traces
           fig.add_trace(
               go.Scatter(x=data["monthyear"], y=data["crimecount"], name="# Crime Incidents"),
               secondary_y=False
           )
           
           fig.add_trace(
               go.Scatter(x=data["monthyear"], y=data["max_snow"], name="Max. Snow (Inches)",line_color='black'),
               secondary_y=True
           )
           
           # Add figure title
           fig.update_layout(
               title_text="<b>Crime </b>and <b>Maximum Snow</b>"
           )
           
           # Set x-axis title
           fig.update_xaxes(title_text="Year")
           
           # Set y-axes titles
           fig.update_yaxes(title_text="<b># Crime Incidents</b>", secondary_y=False)
           fig.update_yaxes(title_text="<b>Max. Snow (Inches)</b>", secondary_y=True)
           fig.update_layout(xaxis=dict(showgrid=False),
                         yaxis=dict(showgrid=False),paper_bgcolor="white",width=int(550),
                         legend=dict(font = dict(size = 8),yanchor="top", y=1.2, xanchor="left", x=.7))
           
           fig['layout']['yaxis2']['showgrid'] = False
           c.plotly_chart(fig)
           
           with col3:
               
               
               c=st.container()
               # Create figure with secondary y-axis
               fig = make_subplots(specs=[[{"secondary_y": True}]])
               
               # Add traces
               fig.add_trace(
                   go.Scatter(x=data["monthyear"], y=data["crimecount"], name="# Crime Incidents"),
                   secondary_y=False
               )
               
               fig.add_trace(
                   go.Scatter(x=data["monthyear"], y=data["max_temperature"], name="Max. Temp. (°C)",line_color='green'),
                   secondary_y=True
               )
               
               # Add figure title
               fig.update_layout(
                   title_text="<b>Crime </b>and <b> Maximum Temperature </b>"
               )
               
               # Set x-axis title
               fig.update_xaxes(title_text="Year")
               
               # Set y-axes titles
               fig.update_yaxes(title_text="<b># Crime Incidents</b>", secondary_y=False)
               fig.update_yaxes(title_text="<b>Max. Temp. (°C)</b>", secondary_y=True)
               fig.update_layout(xaxis=dict(showgrid=False),
                             yaxis=dict(showgrid=False),paper_bgcolor="white",width=int(550),
                             legend=dict(font = dict(size = 8),yanchor="top", y=1.2, xanchor="left", x=.7))
               
               fig['layout']['yaxis2']['showgrid'] = False
               c.plotly_chart(fig)   
               
               
               #graph 
               fig = make_subplots(specs=[[{"secondary_y": True}]])

               # Add traces
               fig.add_trace(
                   go.Scatter(x=data["monthyear"], y=data["crimecount"], name="# Crime Incidents"),
                   secondary_y=False
               )
               
               fig.add_trace(
                   go.Scatter(x=data["monthyear"], y=data["consumer_price_index"], name="CPI",line_color='violet'),
                   secondary_y=True
               )
               
               # Add figure title
               fig.update_layout(
                   title_text="<b>Crime</b> and <b>Consumer Price Index (CPI) </b>"
               )
               
               # Set x-axis title
               fig.update_xaxes(title_text="Year")
               
               # Set y-axes titles
               fig.update_yaxes(title_text="<b># Crime Incidents</b>", secondary_y=False)
               fig.update_yaxes(title_text="<b>CPI</b>", secondary_y=True)
               fig.update_layout(xaxis=dict(showgrid=False),
                             yaxis=dict(showgrid=False),paper_bgcolor="white",
                                width=int(550),
                                     legend=dict(font = dict(size = 8),yanchor="top", y=1.2, xanchor="left", x=.7))
               
               fig['layout']['yaxis2']['showgrid'] = False
               c.plotly_chart(fig)
               
               ## graph 3 
               # Create figure with secondary y-axis
               fig = make_subplots(specs=[[{"secondary_y": True}]])
               
               # Add traces
               fig.add_trace(
                   go.Scatter(x=data["monthyear"], y=data["crimecount"], name="# Crime Incidents"),
                   secondary_y=False
               )
               
               fig.add_trace(
                   go.Scatter(x=data["monthyear"], y=data["max_rain"], name="Max. Rain (Cm)",line_color='orange'),
                   secondary_y=True
               )
               
               # Add figure title
               fig.update_layout(
                   title_text="<b> Crime </b>and <b> Maximum Rain </b>"
               )
               
               # Set x-axis title
               fig.update_xaxes(title_text="Year")
               
               # Set y-axes titles
               fig.update_yaxes(title_text="<b># Crime Incidents</b>", secondary_y=False)
               fig.update_yaxes(title_text="<b>Max. Rain (Cm) </b>", secondary_y=True)
               fig.update_layout(xaxis=dict(showgrid=False),
                             yaxis=dict(showgrid=False),paper_bgcolor="white",
                               width=int(550),
                               legend=dict(font = dict(size = 8),yanchor="top", y=1.2, xanchor="left", x=.7))
               
               fig['layout']['yaxis2']['showgrid'] = False
               c.plotly_chart(fig)