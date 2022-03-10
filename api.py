import uvicorn # used for implementing the server and handling all the calls in Python
import pickle
import sys
from fastapi import FastAPI
from fastapi import Query
from pydantic import BaseModel

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

api = FastAPI()
with open("/Users/heni/OneDrive/EasternUniversity/691capstone/CAPSTONE/APP/APP_KESZ/Api/model/model.pkl", "rb") as f:
    model = pickle.load(f)
@api.get('/')
def index():
    return {'message': "This is the home page of this API. "}
@api.post('/prediction')
def get_target(data:heart):
    received = data.dict()
    a = received['sex']
    b = received['age']
    c = received['BMI']
    d = received['chestpain']
    e = received['chestpressure']
    f = received['diabetes']
    g = received['HBP']
    h = received['angina']
    i = received['stroke']
    j = received['highcholesterol']
    k = received['hightriglyc']
    l = received['sleep_aid']
    m = received['sedatives']
    n = received['vitamins']
    o = received['sleep_hours']
    p = received['sleep_quality']
    q = received['add_salt']
    r = received['meat']
    s = received['fat']
    t = received['eggs']
    v = received['fat_type']
    w = received['alcohol']
    x = received['num_drinks']
    y = received['walking']
    z = received['cardio']
    aa = received['sport']
    bb = received['hobby']
    cc = received['work']
    dd = received['SYS_mean']
    ee = received['DIA_mean']
    ff = received['SERCHOL']
    gg = received['HDLCHOL']
    hh = received['TRIGLYC']

    pred_name = model.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,v,w,x,y,z,aa,bb,cc,dd,ee,ff,gg,hh]]).tolist()[0]
    if pred_name == 1:
        result = "no heart disease"
    else:
        result = "heart disease"

    return {result}

if __name__ == '__main__':
    uvicorn.run(api, host='127.0.0.1', port=8000, debug=True)
