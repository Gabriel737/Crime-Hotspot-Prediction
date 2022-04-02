# Utility functions
import numpy as np
import pandas as pd
import utm
import random
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
import config

def generateRandomCoords(ngbh, geodata):
    """
    Function to generate random location coordinates within a given neighbourhood

    Inputs: ngbh <string>: neighbourhood name
            geodata <GeoPandas DataFrame>: neighbourhood boudaries dataframe
    Output: longitude <float>, latitude <float>
    """
    
    SEARCH_LIMIT = 10

    polygon = geodata[geodata['name'] == ngbh]
    
    if polygon.empty:
        print(ngbh)
        print('Failed to match neighbourhood name!')
        return (0, 0)

    # find the bounds of your geodataframe
    x_min, y_min, x_max, y_max = polygon.geometry.total_bounds
    
    # generate random data within the bounds
    x = np.random.uniform(x_min, x_max, SEARCH_LIMIT)
    y = np.random.uniform(y_min, y_max, SEARCH_LIMIT)

    # convert them to a points GeoSeries
    gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
    
    # only keep those points within polygons
    gdf_points = gdf_points[gdf_points.within(polygon.unary_union)]

    if gdf_points.empty:
        print(ngbh)
        print('Reattempting to retrieve random lat/long coordinate!')
        return generateRandomCoords(ngbh, geodata)
    
    num_candidates = len(gdf_points.index)
    index = random.randint(0, num_candidates - 1)
    
    return gdf_points.iloc[index].coords[0][0], gdf_points.iloc[index].coords[0][1]


def utm2latlong(utm_x, utm_y, utm_zone_no, utm_zone_ltr):
    """
    Function to convert UTM coordinates to latitude and longitude

    Inputs: utm_x <float or array<float>>: UTM Easting coordinate
            utm_y <float or array<float>>: UTM Northing coordinate
            utm_zone_no <int>: UTM zone number
            utm_zone_ltr <string>: UTM zone letter

    Output: longitude <float or array<float>>, latitude <float or array<float>>
    """
    
    lat, long = utm.to_latlon(utm_x, utm_y, utm_zone_no, utm_zone_ltr)
    return long, lat


def getBins(min_,max_,n_bins):
    """
    Function to create bins

    Inputs: min_ <int or float>: starting value
            max_ <int or float>: ending value
            num <int>: number of bins

    Output: bins <array<int> or array<float>>
    """

    bins = np.linspace(start=min_, stop=max_, num=n_bins+1)
    return bins


def getCellLocs(lats,longs,lat_bins,long_bins, correction=False):
    """
    Function to generate cell coordinates

    Inputs: lats <array<float>>: latitude values
            longs <array<float>>: longitude values
            lat_bins <array<float>>: latitude bins
            long_bins <array<float>>: longitude bins
            correction <bool>: if true, the latitude and longitude values outside the bins 
                               would be shifted to the closest bin

    Output: cell coordinate x <int>, cell coordinate y <int>
    """
    
    # Assign x coordinate of cell. X coordinates can range from 1 to lat_bins and are upper bound.
    # Location coordinates laying outside the bounding box are labelled as either 0 or len(lat_bins)
    cell_x = np.digitize(lats,lat_bins,right=True)
    
    # Assign x coordinate of cell. Y coordinates can range from 1 to long_bins and are upper bound.
    # Location coordinates laying outside the bounding box are labelled as either 0 or len(long_bins)
    cell_y = np.digitize(longs,long_bins,right=True)
    
    if correction == True:
        cell_x_corr = [i-1 if i==len(lat_bins) else i+1 if i==0 else i for i in cell_x] 
        cell_y_corr = [i-1 if i==len(long_bins) else i+1 if i==0 else i for i in cell_y]
        
        return cell_x_corr, cell_y_corr
    
    elif correction == False:
        cell_x_excl = [-1 if i==0 or i==len(lat_bins) else i for i in cell_x]
        cell_y_excl = [-1 if i==0 or i==len(long_bins) else i for i in cell_y]
        
        return cell_x_excl, cell_y_excl
    

def getDate(day, month, year):
    """
    Function to fetch crime date from day, month and year

    Inputs: day <int>: day value
            month <int>: month value
            year <int>: year value

    Output: date <datetime>: date
    """

    dt = datetime(year, month, day)
    date = dt.date()
    return date


def getAllCombs(list_1,list_2):
    """
    Function to generate all value combinations of two lists

    Inputs: list_1 <array>: first list
            list_2 <array>: second list

    Output: all_combs <array>: combined list
    """

    all_combs = [(x,y) for x in list_1 for y in list_2]
    return all_combs


def getPivot(data, values, index, columns, aggfunc, n_bins, allcombs=False):
    """
    Function to create a pivot table

    Inputs: data <DataFrame>: unpivoted data
            values <string>: column to aggregate
            index <array<string>>: keys to group by on the pivot table index
            columns <array<string>>: keys to group by on the pivot table column
            aggfunc <function>: aggregate function to get pivot table values
            n_bins <int>: number of latitude/longtitude bins
            all_combs <bool>: if True, all index and column combination 
                              (date and crime category and cell coorindates) are generated

    Output: pivoted data <DataFrame>
    """
    
    # Create a pivot table with cell coordinates as columns and date and crime category as indices
    data_pivot = data.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc)
    
    # Flatten the column values
    data_pivot.columns = data_pivot.columns.to_flat_index()
    
    if allcombs == True:
        
        # All possible cell values along a coordinate
        cell_x_all = np.arange(1,n_bins+1,1)
        
        # Generate all cell cooridinate combinations
        cell_all_pairs = getAllCombs(list_1=cell_x_all, list_2=cell_x_all)
        
        # All unique dates
        unique_dates = data['DATE'].unique()
        
        # Generate all date-crime category combinations
        date_cat_all_pairs = getAllCombs(list_1=unique_dates, list_2=config.CRIME_CATS)
        
        # Reindex the pivot table with all cell coordinate combinations as columns and 
        # all date-crime categories as indices
        data_pivot_ri = data_pivot.reindex(date_cat_all_pairs).reindex(columns=cell_all_pairs).fillna(0)
        
        return data_pivot_ri
    
    else:
        return data_pivot


def getFeaturesTargets(data, seq_len):
    """
    Function to group instances to pairs of batch size and collect the corresponding target sample

    Inputs: data <DataFrame>: input data
            seq_len <int>: number of instances to be considered as a sequence

    Output: features <array>, targets <array>
    """

    features = []
    targets = []
    for i in np.arange(0,data.shape[0]-(seq_len+1)):
        feature_batch = data[i:i+seq_len]
        target = data[i+seq_len+1]
        features.append(feature_batch)
        targets.append(target)
    targets = np.array(targets).sum(axis=1)
    return features, targets

def plotConfusionMatrix(threshold_list, tp_list, fn_list, fn_neigh_pos_list, accuracy_list):

    fig, ax1 = plt.subplots()

    # plot bars in stack manner
    ax1.bar(threshold_list,fn_list,bottom=np.array(fn_neigh_pos_list)+np.array(tp_list), color='red',label='True Positive vs False Negative(%)')
    ax1.bar(threshold_list,fn_neigh_pos_list,bottom=tp_list, color='lightgreen')
    ax1.bar(threshold_list,tp_list,color='g',label='Accuracy(%)')
    ax2 = ax1.twinx()
    ax2.plot(threshold_list,accuracy_list,color='black')

    fig.savefig('../data/eval_plot.png')
    plt.show()