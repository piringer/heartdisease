import streamlit as st
import pandas as pd
import requests
import json
from pydantic import BaseModel

# using Pydantic models to declare request body
# base data as derived from Streamlit Frontend
class heart(BaseModel):
    sex: int
    age: int
    BMI: float
    chestpain: int
    chestpressure: int
    diabetes: int
    HBP: int
    angina: int
    stroke: int
    highcholesterol: int
    hightriglyc: int
    sleep_aid: int
    sedatives: int
    vitamins: int
    sleep_hours: int
    sleep_quality: int
    add_salt: int
    meat: int
    fat: int
    eggs: float
    fat_type: int
    alcohol: int
    num_drinks: int
    walking: int
    cardio: int
    sport: int
    hobby: int
    work: int
    SYS_mean: float
    DIA_mean: float
    SERCHOL: float
    HDLCHOL: float
    TRIGLYC: float

# get input data from streamlit frontend
sex = st.sidebar.number_input("Are you a male or a female? 1:Male, 2:Female", 1,2,1)
age = st.sidebar.number_input("Choose your age group: 1:24 or younger, 2:30-34, 3:35-39, 4:40-44, 5:45-49, 6:50-54, 7:55-59, 8:60 or older", 1,8,1)
BMI = st.sidebar.number_input("What is your BMI?", 1,200,1)
chestpain = st.sidebar.number_input("Do you have chestpain: no=1, yes=2", 1,2,1)
chestpressure = st.sidebar.number_input("Do you have chestpressure:no=1, yes=2", 1,2,1)
diabetes = st.sidebar.number_input("Do you have diabetes? no=1, yes=2", 1,2,1)
HBP = st.sidebar.number_input("Do you have high blood pressure? no=1, yes=2", 1,2,1)
angina = st.sidebar.number_input("Do you have angina pectoris? no=1, yes=2", 1,2,1)
stroke =st.sidebar.number_input("Have you ever had stroke? no=1, yes=2", 1,2,1)
highcholesterol = st.sidebar.number_input("Do you have high cholesterol? no=1, yes=2", 1,2,1)
hightriglyc = st.sidebar.number_input("Do you have high trigyleric? no=1, yes=2", 1,2,1)
sleep_aid = st.sidebar.number_input("How often do you take sleeping aids? 1:every day, 2:a few days a week, 3:once a week, 4:occasionally, 5:rarely, 6:never", 1,6,1)
sedatives = st.sidebar.number_input("How often do you take sedatives? 1:every day, 2:a few days a week, 3:once a week, 4:occasionally, 5:rarely, 6:never", 1,6,1)
vitamins = st.sidebar.number_input("How often do you take vitamins? 1:every day, 2:a few days a week, 3:once a week, 4:occasionally, 5:rarely, 6:never", 1,6,1)
sleep_hours = st.sidebar.number_input("How many hours do you sleep? 1:5 hours or less, 2:6 hours, 3: 7 hours, 4:8hours, 5: 9 hours or more", 1,5,1)
sleep_quality = st.sidebar.number_input("How do you describe the qaulity of your normal sleep? 1:poor, 2:fair, 3:good", 1,3,1)
add_salt = st.sidebar.number_input("How often do you add salt to your food? 1:not at all, 2:sometimes, 3:only after fasting, 4:always", 1,4,1)
meat = st.sidebar.number_input("How often do you eat meat? 1:every day, 2:most days, 3:at least once a week, 4:infrequently, 5:never", 1,5,1)
fat = st.sidebar.number_input("How often do you eat the fat on meat? 1:every day, 2:most days, 3:at least once a week, 4:infrequently, 5:never", 1,5,1)
eggs = st.sidebar.number_input("How many eggs do you eat per week?", 0,100,1)
fat_type = st.sidebar.number_input("Which of the following do you eat most? 1: butter, 2:polyunsaturated margarine, 3:other table margarines, 4:I rarely eat any of these, 5:I don’t eat any of these", 1,5,1)
alcohol = st.sidebar.number_input("How often do you drink alcohol? 1: don’t drink alcohol, 2:less than once a week, 3:on 1 or 2 days a week, 4:on 3 or 4 days a week, 5:on 5 or 6 days a week, 6:every day", 1,6,1)
num_drinks = st.sidebar.number_input("When you drink alcohol, how many drinks do you have per day? 1: don’t drink alcohol, 2:1 or 2 drinks, 3:3 or 4 drinks, 4:5 to 8 drinks, 5:9 to 12 drinks, 6:13 to 20 drinks, 7:more than 20 drinks", 1,7,1)
walking = st.sidebar.number_input("How often do you walk for exercise? 1: three times or more a week, 2:once or twice a week, 3:once a month, 4:rarely, 5:never", 1,5,1)
cardio = st.sidebar.number_input("How often do you do cardio exercise? 1: three times or more a week, 2:once or twice a week, 3:once a month, 4:rarely, 5:never", 1,5,1)
sport = st.sidebar.number_input("How often do you engage in ither sports? 1: three times or more a week, 2:once or twice a week, 3:once a month, 4:rarely, 5:never", 1,5,1)
hobby = st.sidebar.number_input("How often do you engage in other physical activities like hobbies? 1: three times or more a week, 2:once or twice a week, 3:once a month, 4:rarely, 5:never", 1,5,1)
work = st.sidebar.number_input("How much do you spend walking while working? 1: practically all, 2:more than half, 3:about half, 4:less than half, 5:almost none", 1,5,1)
SYS_mean = st.sidebar.number_input("What is your systolic blood pressure?", 1,1000,1)
DIA_mean = st.sidebar.number_input("What is your diastolic blood pressure?", 1,1000,1)
SERCHOL = st.sidebar.number_input("What is your serum cholesterol level?", 1,1000,1)
HDLCHOL = st.sidebar.number_input("What is your HDL cholesterol level?", 1,1000,1)
TRIGLYC = st.sidebar.number_input("What is your triglycerid level?", 1,1000,1)

# combine input to dict
data = {"sex":sex,
        "age":age,
        "BMI":BMI,
        "chestpain":chestpain,
        "chestpressure":chestpressure,
        "diabetes":diabetes,
        "HBP":HBP,
        "angina":angina,
        "stroke":stroke,
        "highcholesterol":highcholesterol,
        "hightriglyc":hightriglyc,
        "sleep_aid":sleep_aid,
        "sedatives":sedatives,
        "vitamins":vitamins,
        "sleep_hours":sleep_hours,
        "sleep_quality":sleep_quality,
        "add_salt":add_salt,
        "meat":meat,
        "fat":fat,
        "eggs":eggs,
        "fat_type":fat_type,
        "alcohol":alcohol,
        "num_drinks":num_drinks,
        "walking":walking,
        "cardio":cardio,
        "sport":sport,
        "hobby":hobby,
        "work":work,
        "SYS_mean":SYS_mean,
        "DIA_mean":DIA_mean,
        "SERCHOL":SERCHOL,
        "HDLCHOL":HDLCHOL,
        "TRIGLYC":TRIGLYC
}

# create json object from dictionary
dataJSON = json.dumps(data)

# make prediction by making post request to the API
pred = requests.post(url = "http://127.0.0.1:8000/prediction/", data=dataJSON) # heartapi

# display prediction
st.image("/Users/heni/OneDrive/EasternUniversity/691capstone/CAPSTONE/APP/APP_KESZ/App//heart1.png", use_column_width=True)
st.write('Please fill out the form')
if st.button('Submit'):
     st.subheader("Based on provided data, patient probably has ".format(pred.json()))
     st.subheader("{}.".format(pred.json()))

if st.checkbox("Show Details"):
    st.write("The data retreived from Streamlit's Frontend: the user input is converted into a JSON object and sent to an endpoint of an API that is built with the FastAPI Framework, the `/prediction/` endpoint. The input data looks as follows:")
    st.write(data)
    st.write("This endpoint returns the prediction:")
    st.write(pred.json())
