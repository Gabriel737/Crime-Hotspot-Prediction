# Crime-Hotspot-Prediction

## Project Objective
This project aims to identify areas and neighbouthoods with elevated crime risk in major Canadian cities (presently Vancouver) using a spatial-temporal neural network architecture. 

![WhatsApp Image 2022-04-10 at 10 30 56 PM](https://user-images.githubusercontent.com/26691915/162671625-5cc9d3a2-7ad4-4064-85bd-9859c03481a1.jpeg)

## Datasets Used
* Vancouver Police Department Open Data on crime incidents ([source](https://geodash.vpd.ca/opendata/))
* Vancouver local neighbourhood boundary data ([source](https://opendata.vancouver.ca/explore/dataset/local-area-boundary/information/?disjunctive.name&location=11,49.2474,-123.12402))
* Secondary data source for extended insights eg. housing price index, consumer price index, weather etc.

**Note**: Input features and target data files can be downloaded from [this link](https://drive.google.com/drive/folders/1n4d247P9sBAOvWwQL6S-VYkw63mQNSaO?usp=sharing) and need to be pasted in the `data\processed` folder.

## Project Pipeline
![WhatsApp Image 2022-04-10 at 10 45 13 PM](https://user-images.githubusercontent.com/25038038/162674753-05b25cb7-39c9-4588-a078-fb5bdf9a4373.jpeg)

## Demo

Install the required dependencies:

``` pip install requirements.txt ```

Go to app folder and run: 

``` streamlit run main.py ```

This will run the streamlit web application on localhost.


## Model Training and Testing

Install the required dependencies:

``` pip install requirements.txt ```

Go to code folder and run:

``` python train.py ```

This would train the model with the parameter values mentioned in `config.py`.

