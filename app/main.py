#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:39:04 2022

@author: arshdeepsingh
"""



from hydralit import HydraApp
import streamlit as st
from validation import Validation
from prediction import Prediction
from eda import EDA
import hydralit_components as hc


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        
        padding-top: 20px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# REMOVING TOP PADDINGS AND MARGINS
st.markdown("""
        <style>
                .css-18e3th9 {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# REMOVING HAMBURGER MENU DEFAULT BUTTON (RIGHT SIDE), AND FOOTER
st.markdown(""" <style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)




if __name__ == '__main__':

    
    theme_neutral = {'menu_background':'black'}

    #this is the host application, we add children to it and that's it!
    app = HydraApp(title='SFUMLites',favicon="🐙",navbar_theme=theme_neutral,navbar_animation=True)
  
    #add all your application classes here
    app.add_app("Prediction Map", app=Prediction())
    app.add_app("KPIs", app=Validation())
    app.add_app("Trends", app=EDA())

    #run the whole lot
    app.run()