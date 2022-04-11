#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:40:52 2022

@author: arshdeepsingh
"""

import streamlit as st
import pandas as pd
import numpy as np

#add an import to Hydralit
from hydralit import HydraHeadApp


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
from sklearn.metrics import f1_score,plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
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


def add_categorical_legend(folium_map, title, colors, labels):
    
    
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map
#create a wrapper class
class Validation(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        # st.write('eval')
        # Create a page dropdown 
                
        st.markdown("<h2 style='text-align: Left; color: black;'>Hotspot Prediction KPIs</h2>", unsafe_allow_html=True)
        # st.title("Evaluation Dashboard")
        startdate=datetime(2020, 1, 17)
        enddate=datetime(2022,2,9)
        
        st.sidebar.markdown(f'<div style="color:black;text-align: center;font-size:16px;"><b>{"Understand how good the results are? "}</b></div>', unsafe_allow_html=True)
        desc='<br> Choose a date for which the predicted hotspots need to be evaluated using the field below and click on Predict. The colour coded results are displayed on an interactive map along with some important KPIs. <br> Note: FN stands for False Negatives. <br>  <br> <b>Appendix:</b> <br><b>Recall -</b> % hotspots correctly predicted <br> <b>Precision - </b>% correctly predicted hotspots <br> <b>F1 Score -</b> combination of precision and recall <br> <b>True Positives -</b> No. of correctly identified hotspots <br> <b>False Negatives with Positive Neighbours -</b> No. of wrongly classified safe areas with hotspot predicted in vicinity <br> <b>False Negatives with Negative Neighbours -</b> No. of wrongly classified safe areas with no hotspot predicted in vicinity'
        st.sidebar.markdown(f'<div style="color:black;text-align: left;font-size:12px;justify-content:justify">{desc}</div>', unsafe_allow_html=True)
        st.sidebar.write('\n')
        st.sidebar.write('\n')

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
        features=read_h5('data/processed/features.h5','test')
        targets=read_h5('data/processed/targets.h5','test')
        sec_features=read_h5('data/processed/sec_features.h5','test')

        if submit:
            # col1, col2, col3 = st.columns([5,1,5])
            
            cont1=st.container()
            col11,col12,col13,col14 = cont1.columns([2,3,3,7])

                    
            # import streamlit as st

            from load_css import local_css
            
            local_css("style.css")
             
            t_style = "<span style='color:black;text-align:center;font-size:11px;' class='highlight green'><span class='bold'>True Positive</span></span>"
            t_style2 = "<span style='color:black;text-align:center;font-size:11px;' class='highlight blue'><span class='bold'>FN With Negative Neighbour</span></span>"
            t_style3 = "<span style='color:black;text-align:center;font-size:11px;' class='highlight red'><span class='bold'>FN With Positive Neighbour</span></span>"
            
            col11.markdown(t_style, unsafe_allow_html=True) 
            col12.markdown(t_style2, unsafe_allow_html=True) 
            col13.markdown(t_style3, unsafe_allow_html=True) 
            
            
            col1, col2, col3,col4 = st.columns([6,0.5,4,3])
            
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
            
            # st.write(str(delta))
            #Import libraries
            X_test=features[delta]
            X_test=np.expand_dims(X_test,0)
            
            y_test=targets[delta]
            y_test=np.expand_dims(y_test,0)
            X_sec_test=sec_features[delta]
            X_sec_test=X_sec_test.reshape(1,-1)
            
            X_sec_test=torch.from_numpy((X_sec_test)).float()

            # st.write(y_test.shape)
            X_test=torch.from_numpy((X_test)).float()
            y_test=torch.from_numpy((y_test)).float()
            y_test=y_test.view(26,26)
            
            y_test = (y_test > 0).int()
            
            model.eval()
            y_pred = model(X_test,X_sec_test)
            # st.write(y_pred.shape)
            y_pred=y_pred.view(26,26)
            

            y_pred=y_pred.view(26,26)
            y_pred_bin = (y_pred > config.CLASS_THRESH).float()
            # st.write(y_pred_bin.sum())
            
            arr=y_pred_bin.cpu().detach().numpy()
            
            (x,y)=np.where(arr == 1)
            
            arr_prob=y_pred.cpu().detach().numpy()
            
            arr_test=y_test.cpu().detach().numpy()
            
            # st.write((str(np.unique(arr_test))))
            
            # st.write(f1_score(arr_test.flatten(),arr.flatten()))
            
            
            (x1,y1)=np.where(arr_test == 1)
            
            FN_with_pos_x=[]
            
            FN_with_neg_x=[]
            
            TP_x=[]
            
            FN_with_pos_y=[]
            
            FN_with_neg_y=[]
            
            TP_y=[]
            tp,tn,fn,fp=0,0,0,0
            
            for j in range(0,arr.shape[0]):
              for k in range(0,arr.shape[1]):
                 
                if(arr[j][k]==0 and arr_test[j][k]==1): 
                    fn+=1
                    
                if(arr[j][k]==1 and arr_test[j][k]==1): 
                    tp+=1
                    
                if(arr[j][k]==1 and arr_test[j][k]==0): 
                    fp+=1
                    
                if(arr[j][k]==0 and arr_test[j][k]==0): 
                    tn+=1
                     
                if(arr[j][k]==1 and arr_test[j][k]==1): 
                    TP_x.append(j)
                    TP_y.append(k)
                    
                if(arr[j][k]==0 and arr_test[j][k]==1):
                  n1=arr[j-1][k] if (j-1)>=0 else 0
                  n2=arr[j][k-1] if (k-1)>=0 else 0
                  n3=arr[j+1][k] if (j+1)<arr.shape[0] else 0
                  n4=arr[j][k+1] if (k+1)<arr.shape[1] else 0
                  n5=arr[j-1][k-1] if (j-1)>=0 and (k-1)>=0  else 0  
                  n6=arr[j+1][k-1] if (j+1)<arr.shape[0] and (k-1)>=0  else 0
                  n7=arr[j-1][k+1] if (j-1)>=0 and (k+1)<arr.shape[1]  else 0
                  n8=arr[j+1][k+1] if (j+1)<arr.shape[0] and (k+1)<arr.shape[1]  else 0
                  
                  if(n1+n2+n3+n4+n5+n6+n7+n8>=1):
                      
                    FN_with_pos_x.append(j)
                    FN_with_pos_y.append(k)
                  else:
                    FN_with_neg_x.append(j)
                    FN_with_neg_y.append(k)
            
            prec=tp/(tp+fp)
            recall=tp/(tp+fn)
            f1=2*(prec*recall)/(prec+recall)
            
            # ##DISPLAY MAP for predicted
            with col1:
            
                m = folium.Map(location=[(min_lat+max_lat)/2, (min_long+max_long)/2],zoom_start=6,tiles='OpenStreetMap') #Create a empty folium map object
                m.fit_bounds([[min_lat,min_long], [max_lat,max_long]])
                folium.Rectangle(bounds=[(min_lat,min_long),(max_lat,max_long)], color='yellow', fill=True, fill_color='#ffffff', fill_opacity=0.1).add_to(m)
                
                
                
                for i in range(len(TP_x)):
                    folium.Rectangle(bounds=[(lat_bins[TP_x[i]],long_bins[TP_y[i]]),(lat_bins[TP_x[i]+1],long_bins[TP_y[i]+1])], color='green', fill=True, fill_color='green', fill_opacity=0.8,stroke=False).add_to(m)
                    
                for i in range(len(FN_with_pos_x)):
                    folium.Rectangle(bounds=[(lat_bins[FN_with_pos_x[i]],long_bins[FN_with_pos_y[i]]),(lat_bins[FN_with_pos_x[i]+1],long_bins[FN_with_pos_y[i]+1])], color='blue', fill=True, fill_color='blue', fill_opacity=0.8,stroke=False).add_to(m)
                
                for i in range(len(FN_with_neg_x)):
                    folium.Rectangle(bounds=[(lat_bins[FN_with_neg_x[i]],long_bins[FN_with_neg_y[i]]),(lat_bins[FN_with_neg_x[i]+1],long_bins[FN_with_neg_y[i]+1])], color='red', fill=True, fill_color='red', fill_opacity=0.8,stroke=False).add_to(m)
                  
                m = add_categorical_legend(m, 'My title',
                             colors = ['#000','#03cafc'],
                           labels = ['Heat', 'Cold'])
                folium_static(m,width=520,height=380,)
                

                
                with col3:
                    # st.write(tp)
                    # st.write(len(FN_with_neg_x))
                    # st.write(len(FN_with_pos_x))
                    # st.write(recall)
                    
                    # variables
                    wch_colour_box = (0, 204, 102)  # green
                    wch_colour_font = (0, 0, 0)  # blackfont
                    fontsize = 35
                    valign = "left"
                    iconname = "fas fa-asterisk"
                    # sline = "Recall" # kpi name
                    lnk = ''
                    # i = 26  # kpi value
                    
                    def disp(value,heading):
                        sline=heading
                        i=value
                        htmlstr = f"""<p style='background-color:white;
                                    font-face:Nexa Bold;
                                    color: black;
                                    font-size: {fontsize}px;
                                    border-radius: 10px;
                                    padding-left: 5px;
                                    padding-right: 35px;
                                    text-align:center;
                                    padding-top: 0px;
                                    padding-bottom: 5px;
                                    line-height:30px;'>
                                    </style> <span style='font-size: 15px; font-family: "Times New Roman";
                                     margin-top: 0;'>{sline}</style>
                                     </span><BR>
                                   <b>{i}%</b></p>"""
                                   
                        st.markdown(lnk + htmlstr, unsafe_allow_html=True)

                                   
                    def disp1(value,heading):
                        sline=heading
                        i=value
                        htmlstr = f"""<p style='background-color:white;
                                    font-face:Nexa Bold;
                                    color: black;
                                    font-size: {fontsize}px;
                                    border-radius: 10px;
                                    padding-left: 5px;
                                    padding-right: 35px;
                                    text-align:center;
                                    padding-top: 0px;
                                    padding-bottom: 5px;
                                    line-height:30px;'>
                                    </style> <span style='font-size: 15px; font-family: "Times New Roman";
                                     margin-top: 0;'>{sline}</style>
                                     </span><BR>
                                   <b>{i}</b></p>"""
                                   
                        st.markdown(lnk + htmlstr, unsafe_allow_html=True)
                        
                        
                    disp(int(f1*100),'F1 Score')
                    disp(int(recall*100),'Recall')
                    disp(int(prec*100),'Precision')
                    
                    # disp1(int(tp),'True Positives')
                    # disp1(int(len(FN_with_neg_x)),'FN with Negative Neighbours')
                    # disp1(int(len(FN_with_pos_x)),'FN with Positive Neighbours')
                    
                    # st.write('ghey')
                    # st.metric('Recall',str(int(recall*100))+'%' )
                with col4:
                    wch_colour_box = (0, 204, 102)  # green
                    wch_colour_font = (0, 0, 0)  # blackfont
                    fontsize = 35
                    valign = "left"
                    iconname = "fas fa-asterisk"
                    # sline = "Recall" # kpi name
                    lnk = ''
                    # i = 26  # kpi value
                    
                    def disp(value,heading):
                        sline=heading
                        i=value
                        htmlstr = f"""<p style='background-color:white;
                                    font-face:Nexa Bold;
                                    color: black;
                                    font-size: {fontsize}px;
                                    border-radius: 10px;
                                    padding-left: 5px;
                                    padding-right: 35px;
                                    text-align:center;
                                    padding-top: 0px;
                                    padding-bottom: 5px;
                                    line-height:30px;'>
                                    </style> <span style='font-size: 15px; font-family: "Times New Roman";
                                     margin-top: 0;'>{sline}</style>
                                     </span><BR>
                                   <b>{i}%</b></p>"""
                                   
                        st.markdown(lnk + htmlstr, unsafe_allow_html=True)

                                   
                    def disp1(value,heading):
                        sline=heading
                        i=value
                        htmlstr = f"""<p style='background-color:white;
                                    font-face:Nexa Bold;
                                    color: black;
                                    font-size: {fontsize}px;
                                    border-radius: 10px;
                                    padding-left: 5px;
                                    padding-right: 35px;
                                    text-align:center;
                                    padding-top: 0px;
                                    padding-bottom: 5px;
                                    line-height:30px;'>
                                    </style> <span style='font-size: 15px; font-family: "Times New Roman";
                                     margin-top: 0;'>{sline}</style>
                                     </span><BR>
                                   <b>{i}</b></p>"""
                                   
                        st.markdown(lnk + htmlstr, unsafe_allow_html=True)
                        
  
                    
                    disp1(int(tp),'True Positives')
                    disp1(int(len(FN_with_neg_x)),'FN with Negative Neighbours')
                    disp1(int(len(FN_with_pos_x)),'FN with Positive Neighbours')
                    
                    # st.write('ghey')
                    # st.metric('Recall',str(int(recall*100))+'%' )
                
            