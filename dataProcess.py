# 데이터처리를 위한 패키지
import pandas as pd
import numpy as np

# 날짜 처리를 위한 패키지
import datetime as dt

# 파일 위치 이름 가져오기 위한 패키지
import glob

# encoding 문제 해결을 위한 패키지
import chardet

#### 데이터 처리를 위한 함수들 ####
def rainOrNot(x):
    '''
    비가 오면 1, 안오면 0
    수치형으로 return
    '''
    if x>0:
        return 1
    else:
        return 0

def snowOrNot(x):
    '''
    눈이 오면 1, 안오면 0
    수치형으로 return
    '''
    if x>0:
        return 1
    else:
        return 0

def pm10Cat(pm_10):
    '''
    https://bluesky.seoul.go.kr/finedust/common-sense/page/10?article=745
    미세먼지(pm10) : 단위 ㎍/㎥ 
    0~3 : 좋음(0) ~ 매우나쁨(3)
    수치형으로 return
    '''
    if pm_10 <= 30:
        return 0
    elif(pm_10>30)&(pm_10<=80):
        return 1
    elif(pm_10>80)&(pm_10<=100):
        return 2
    else:#(pm10>=100)
        return 3

def pm25Cat(pm_25):
    '''
    사실은 25가 아니라 2.5
    즉, 초미세먼지
    https://bluesky.seoul.go.kr/finedust/common-sense/page/10?article=745
    초미세먼지(pm2.5) : 단위 ㎍/㎥ 
    0~3 : 좋음(0) ~ 매우나쁨(3)
    수치형으로 return
    '''
    if(pm_25<15):
        return 0
    elif(pm_25>=15)&(pm_25<35):
        return 1
    elif(pm_25>=35)&(pm_25<75):
        return 2
    else: #(pm25>=75)
        return 3
    
def pmAllCat(x):
    '''
    미세먼지나, 초미세먼지의 값에 따라 그날의 공기상태를 돌려준다.
    수치형으로 return
    '''
    if(x==0):
        return 0
    elif(x==1):
        return 1
    elif(x==2):
        return 2
    else:
        return 3
    
#### uv데이터를 위한 함수 ####
def yyyy(year):
    year_str='{:0>4}'.format(year) # 년도 4자리의 형태로
    return year_str

def mm(month):
    mm='{:0>2}'.format(month) # 월 2개의 자리수로(자리수가 부족하면 숫자0으로 자리 메꾸기, 예: 1월 => 1 => 01
    return mm

def dd(day):
    dd='{:0>2}'.format(day)
    return dd


# 원본 파일 위치
# D:/project/contest/data/original/uv/
#                                      201606, 201607...의 규칙으로 저장됨
# glob을 위한 전용 함수 : 각 년도 폴더의 csv자료만 검색 리스트로 리턴
def filePath(year, month, path='D:/project/contest/data/original/uv/'):
    '''
    glob를 쓸때, 각 폴더안에 있는 uv csv자료를 리스트를 가져오고 싶은데 하나하나 치기 귀찮아서, 
    자료를 다운 받을때 썼던 폴더명 규칙에 따라 필요한 경로를 만들도록 제작한 함수.
    
    자료폴더구조 : path/년도/년도월/csv자료
    
    filePath(2016, 7, path='D:/project/contest/data/original/uv/')
    >> 'D:/project/contest/data/original/uv/2016/201607/*.csv'
    '''
    file = path+yyyy(year)+'/'+yyyy(year)+mm(month)+'/*.csv'
    return file

def find_encoding(fname):
    '''
    안타갑게도, 자료 관리자가 각 csv파일의 encoding을 멋대로 저장했기 때문에
    하나하나 찾기엔 시간이 너무 걸려, 자동으로 encoding을 찾아서 돌려주는 함수
    '''
    # 원하는 파일을 연다
    r_file = open(fname, 'rb').read()
    # chardet을 이용하여, encoding정보 추출
    result = chardet.detect(r_file)
    charenc = result['encoding']
    # encoding정보 str으로 리턴
    return charenc

#### social 데이터를 위한 함수 ####
def changeColNames(df, before, after) : 
    '''
    dataframe을 받아, 컬럼명에 원하는 문자열을 수정.
    모든 데이터들의 날짜컬럼 이름을 date로 통일
    before : column명의 수정하고 싶은 문자열
    after : 대체할 문자열
    '''
    new_col_names = ['date']
    new_col_names.extend(list(df.columns)[1:]) # 모든 소셜데이터의 날자는 첫번째 column에 위치
    df.columns = new_col_names
    return pd.Series(df.columns).apply(lambda x : x.replace(before,after))
