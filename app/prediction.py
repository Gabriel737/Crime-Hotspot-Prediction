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
import torch
import h5py
from model import HotspotPredictor
import streamlit.components.v1 as components


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

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
class Prediction(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        
        #-------------------existing untouched code------------------------------------------
   
       st.sidebar.markdown(f'<div style="color:black;text-align: center;font-size:16px;"><b>{"Find out future risky hotspots?"}</b></div>', unsafe_allow_html=True)
       desc='<br> Choose a specific date for which hotspots need to be identified using the field below and click on Predict. The results are displayed an interactive map along with the list of risky neighbourhoods'
       st.sidebar.markdown(f'<div style="color:black;text-align: justify;font-size:12px;justify-content:justify">{desc}</div>', unsafe_allow_html=True)
       st.sidebar.write('\n')
       st.sidebar.write('\n')

       # components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
       # st.sidebar.write(":heavy_minus_sign:" * 34)
       # Create a page dropdown 
       
       # cont=st.container()
       
       cont1=st.container()
       col11,col12,col13= cont1.columns([4,1,3])
       
       col11.markdown("<h2 style='text-align: Left; color: black;'>Hotspot Prediction Map</h2>", unsafe_allow_html=True)
       
       col13.markdown("<h2 style='text-align: Left; color: black;'>Risky Crime Areas</h2>", unsafe_allow_html=True)

       
       # st.title("Crime Prediction Dashboard")
       startdate=datetime(2020, 1, 17)
       enddate=datetime(2022,2,9)
       date_global = st.sidebar.date_input("Choose a date",  min_value=startdate, max_value=enddate,value=startdate)
       
       # st.write(date_global)
       areas=[]
       prob=[]
       geodata = gpd.read_file('local-area-boundary.shp')
       def get_neighbourhoods(x_min, y_min, x_max, y_max, geodata):
       
           square_border = LineString([(x_min, y_min), (x_min, y_max), 
                               (x_min, y_max), (x_max, y_max), 
                               (x_max, y_max), (x_max, y_min), 
                               (x_max, y_min), (x_min, y_min)])
       
           square = Polygon(square_border)
       
           result = geodata['geometry'].intersects(square)
           result = geodata.merge(result.rename('intersects'), left_index=True, right_index=True)
           result = result[result['intersects'] == True]
           result = result[['name']]   
           return result
       
       with st.sidebar:
           cola1,cola2,cola3=st.sidebar.columns([1,5,1])
           cola1,cola2,cola3=st.columns(3)
           
           
           with cola2:
               submit = cola2.button("Predict")
               
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               st.write("")
               
               image = Image.open('logo.svg.png')
               cola2.image(image, caption='')

       vpd='Vancouver Police Department'
       st.sidebar.markdown(f'<div style="color:black;text-align: center;font-size:13px;justify-content:justify">{vpd}</div>', unsafe_allow_html=True)
               
    
       date_min=datetime.strptime('17-01-2020','%d-%m-%Y')
       date_max=datetime.strptime('11-02-2022','%d-%m-%Y')
       d='cpu'
       device = torch.device(d)
       model =  HotspotPredictor(input_dim=len(config.CRIME_CATS), hidden_dim=config.HIDDEN_DIM, kernel_size=config.KERNEL_SIZE, bias=True)
       model.to(device)
       model_path='data/model_states/best_model_optim-(Adam)_lr-(3e-05)_bs-(32)_thres-(0.5)_rs-(42)-nepoch-(60)_wcew-([1, 3].pt'
       load_model(model_path,model)
       ##secondary features
       features=read_h5('data/processed/features.h5','test')
       targets=read_h5('data/processed/targets.h5','test')
       # sec_features = pd.read_csv('data/processed/cpi_hpi_weather_data.csv',index_col=0).values
       sec_features=read_h5('data/processed/sec_features.h5','test')

       if submit:
     
           col1, col2, col3 = st.columns([6,0.5,4])
           
           date_global=datetime.strptime(str(date_global),'%Y-%m-%d')
           delta = (date_global - date_min).days+1
                
           # Number of bins = bounding box length/ length of each cell
           n_bins = int(config.BB_DIST/config.BB_CELL_LEN)
               
           # Get minimum and maxiumum values for longitudes and latitudes of bounding box vertices
           min_lat = min(config.BB_VERTICES.values(), key = lambda x: x['lat'])['lat']
           max_lat = max(config.BB_VERTICES.values(), key = lambda x: x['lat'])['lat']
           
           min_long = min(config.BB_VERTICES.values(), key = lambda x: x['long'])['long']
           max_long = max(config.BB_VERTICES.values(), key = lambda x: x['long'])['long']
           
           # Divide bounding box into bins
           lat_bins = getBins(min_=min_lat, max_=max_lat, n_bins=n_bins)
           long_bins = getBins(min_=min_long, max_=max_long, n_bins=n_bins)
           
           ##DISPLAY MAP
           with col1:
               # st.image(image, caption='',width=220,use_column_width='never')
               # st.write(str(delta))
               #Import libraries
               X_test=features[delta]
               X_test=np.expand_dims(X_test,0)
               
               X_sec_test=sec_features[delta]
               X_sec_test=X_sec_test.reshape(1,-1)
               # st.write(str(X_sec_test.shape))
               
               y_test=targets[delta]
               y_test=np.expand_dims(y_test,0)
               # st.write(y_test.shape)
               X_test=torch.from_numpy((X_test)).float()
               
               
               X_sec_test=torch.from_numpy((X_sec_test)).float()
               
               y_test=torch.from_numpy((y_test)).float()
               y_test=y_test.view(26,26)
               
               model.eval()
               # st.write(X_test.shape)
               # st.write(len(sec_features.columns))
               # st.write(len(sec_features))
               # st.write('debug')
               # st.write(str(X_test.shape))
               # st.write(str(X_sec_test.shape))
               
               y_pred = model(X_test,X_sec_test)
               
               # st.write('debug1')
               # st.write(y_pred.shape)
               y_pred=y_pred.view(26,26)
               
    
               y_pred=y_pred.view(26,26)
               y_pred_bin = (y_pred > config.CLASS_THRESH).float()
               # st.write(y_pred_bin.sum())
               
               arr=y_pred_bin.cpu().detach().numpy()
               
               (x,y)=np.where(arr == 1)
               
               arr_prob=y_pred.cpu().detach().numpy()
               
               arr_test=y_test.cpu().detach().numpy()
               
               (x1,y1)=np.where(arr_test == 1)
               
               
               m = folium.Map(location=[(min_lat+max_lat)/2, (min_long+max_long)/2],zoom_start=6,tiles='OpenStreetMap') #Create a empty folium map object
               m.fit_bounds([[min_lat,min_long], [max_lat,max_long]])
               folium.Rectangle(bounds=[(min_lat,min_long),(max_lat,max_long)], color='yellow', fill=True, fill_color='#ffffff', fill_opacity=0.1).add_to(m)
               
               for i in range(len(x)):
                   folium.Rectangle(bounds=[(lat_bins[x[i]],long_bins[y[i]]),(lat_bins[x[i]+1],long_bins[y[i]+1])], color='red', fill=True, fill_color='red', fill_opacity=0.5,stroke=False).add_to(m)
               
               st.write("")
               # call to render Folium map in Streamlit
               folium_static(m,width=600,height=380)
               
           ##DISPLAYS TABLE FOR AREA NAMES   
           with col3:
               
               
               # x_min=lat_bins[x[i]]
               # x1=lat_bins[x[i]]
               # x2=lat_bins[x[i]+1]
               # x3=lat_bins[x[i]+1]
               # y0=long_bins[y[i]]
               # y1=long_bins[y[i]+1] 
               # y2=long_bins[y[i]]
               # y3=long_bins[y[i]+1]
               
               for i in range(len(x)):
                   # folium.Rectangle(bounds=[(lat_bins[x[i]],long_bins[y[i]]),(lat_bins[x[i]+1],long_bins[y[i]+1])], color='red', fill=True, fill_color='red', fill_opacity=0.1).add_to(m)
                   # st.write(str(i))
                   # st.write(str(lat_bins[x[i]]),str(lat_bins[x[i]+1]))
                   # st.write(str(long_bins[y[i]]),str(long_bins[y[i]+1]))
                   # st.write()
                   t = get_neighbourhoods(lat_bins[x[i]],lat_bins[x[i]+1],long_bins[y[i]],long_bins[y[i]+1], geodata)
                   if(t is not None):
                   #     st.table(t)
                       areas=areas+t.name.tolist()
                       # st.write(str(round(arr_prob[x[i],y[i]],2)),'---',i)
                       tem=str(round((arr_prob[x[i],y[i]]),2))
                       # st.write(tem)
                       prob.append(tem)
          
               areas=list(set(areas))
               # st.write(str(type(prob)))
             
               df=pd.DataFrame(list(zip(areas,prob)))
               df.columns=['Predicted Areas','Risk Probability']
               # df.columns=df.columns.str.title()
              
               styler = df.style.hide_index()
               
               hide_dataframe_row_index = """
               <style>
               .row_heading.level0 {display:none}
               .blank {display:none}
               </style>
               """
    
               # Inject CSS with Markdown
               st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
               #st.table()
               
     # creating table with AgGrid
     
               # cellsytle_jscode = JsCode(
               #     """
               # function(params) {
               #     if (params.value >= 0.6) {
               #         return {
               #             'color': 'black',
               #             'backgroundColor': 'white'
               #         }
               #     } else {
               #         return {
               #             'color': 'black',
               #             'backgroundColor': 'white'
               #         }
               #     }
               # };
               # """
               # )
               
               
             
               gb = GridOptionsBuilder.from_dataframe(df)
               gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
               gb.configure_side_bar(columns_panel=True) #Add a sidebar
               #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
               gb.configure_column("Risk Probability")
               gridOptions = gb.build()
    
               AgGrid(
                   
                      df,
                      theme='fresh',
                      height=380,
                      width=230,
                      gridOptions=gridOptions,
                      fit_columns_on_grid_load=False,
                      enable_enterprise_modules=True,
                      allow_unsafe_jscode=True
                      
                      )