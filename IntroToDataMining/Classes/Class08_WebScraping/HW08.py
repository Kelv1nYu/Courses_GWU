#%%

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from bs4 import BeautifulSoup
import re
import pandas as pd

#%%

def goPage(zip):
    inp = driver.find_element_by_css_selector('input#inputstring')
    sleep(0.1)
    inp.clear()
    sleep(0.1)
    inp.send_keys(zip)
    sleep(1.0)
    inp.send_keys(Keys.DOWN)
    sleep(0.1)
    driver.find_element_by_css_selector('input#btnSearch').click()
    sleep(3)
    # driver.refresh()
    return driver

def getGovWeatherTemperature(soup):
    selectTemp = soup.select('div#current_conditions-summary p.myforecast-current-lrg')
    return (selectTemp[0].text) if (len(selectTemp) == 1) else "error"

def getGovWeatherDateTime(soup):
    selectDateTime = soup.select('div#current_conditions_detail td')
    return selectDateTime[-1].text.strip()

def getGovWeatherLatLongElve(soup):
    selectInfo = soup.select('span.smallTxt')
    strInfo = selectInfo[0].text
    pattern = re.compile(r'\d+.\d+°[A-Z]')
    Info = pattern.findall(strInfo)
    latInfo = Info[0]
    longInfo = Info[1]
    elve = re.compile(r'\d+[a-z]+')
    elveInfo = elve.findall(strInfo)[0]
    return latInfo, longInfo, elveInfo
    # Lat = str(selectLat[0].text).split(" ")


def loadData(zips):
    myTemp = pd.DataFrame(columns=['zip','temperature','datetime','lat','long','elevation'])
    for zipCode in zips:
        driver = goPage(zipCode)
        soup = BeautifulSoup(driver.page_source, 'html5lib')
        tmp = getGovWeatherTemperature(soup)
        dT = getGovWeatherDateTime(soup)
        lt, lg, ev = getGovWeatherLatLongElve(soup)
        
        myTemp = myTemp.append({'zip': zipCode, 'temperature': tmp, 'datetime': dT, 'lat': lt, 'long': lg, 'elevation': ev}, ignore_index=True)

    return myTemp

        

        



#%%

zips = [90210, 20052, 20051, 20050, 20001, 20005, 20009, 20013, 20018, 20023, 20029, 20035, 20039, 20044]
driver = webdriver.Chrome(r'/Users/yukaiqi/Downloads/chromedriver')
driver.get("https://www.weather.gov")

df = loadData(zips)
df.to_csv("mytemp.csv")
driver.quit()






# %%
