#Facebook SC
# selenium-related
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from lxml import etree 
from collections import OrderedDict
import shutil

# other necessary ones
import urllib.request
import pandas as pd
import json
import time
import re
import datetime
from flask import Flask, request, render_template, make_response, session,jsonify,send_from_directory,send_file
from werkzeug.utils import secure_filename
import numpy as np
import os
from wordcloud import WordCloud # ใช้ทำ Word Cloud
import matplotlib.pyplot as plt # ใช้ Visualize Word Cloud
from pythainlp.tokenize import word_tokenize # เป็นตัวตัดคำของภาษาไทย


#dash
import schedule
import time
import subprocess
# from dash import dash, dcc, html, Input, Output,dash_table
# import plotly
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import plotly.express as px
# from werkzeug.middleware.dispatcher import DispatcherMiddleware
# from werkzeug.serving import run_simple


# ทำความสะอาด
from attacut import tokenize, Tokenizer
import pythainlp
from pythainlp import sent_tokenize, word_tokenize, Tokenizer
from pythainlp.util import normalize
from pythainlp.corpus.common import thai_words
from collections import OrderedDict
from sklearn.cluster import KMeans
from pythainlp import spell
from pythainlp.spell import NorvigSpellChecker
from pythainlp.corpus import ttc
from pythainlp import correct
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer,util
import ast
import nltk
nltk.download('punkt')
nltk.download(['punkt_tab', 'wordnet', 'omw-1.4', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from textblob import TextBlob
import secrets
# model แยกความคิดเห็น
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import AutoTokenizer
class TextClassificationDataset(Dataset):
  def __init__(self, texts, labels, tokenizer_Aut, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer_Aut = tokenizer_Aut
        self.max_length = max_length
  def __len__(self):
      return len(self.texts)
  def __getitem__(self, idx):
      text = self.texts[idx]
      label = self.labels[idx]
      encoding = self.tokenizer_Aut(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
      return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
class BERTClassifier(nn.Module):
  def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
  def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
def predict_sentiment_eng(text, model_eng, tokenizer_Autp_E, device, max_length=128):
    model_eng.eval()
    encoding = tokenizer_Autp_E(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = eval(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    if preds.item() == 1:
      L = "tell experience"
    elif preds.item() == 2:
      L = "Question"
    else:
      L = "useless/unimportant"
    return L
def predict_sentiment_thai(text, model_thai, tokenizer_Autp_T, device, max_length=128):
    model_thai.eval()
    encoding = tokenizer_Autp_T(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model_thai(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    if preds.item() == 1:
      L = "tell experience"
    elif preds.item() == 2:
      L = "Question"
    else:
      L = "useless/unimportant"
    return L
# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
# Model eng
bert_model_name_eng = 'bert-large-uncased'
model_eng = BERTClassifier(bert_model_name_eng, num_classes).to(device)
model_eng.load_state_dict(torch.load("bert_classifier_eng.pth", map_location=device))
tokenizer_Autp_E = AutoTokenizer.from_pretrained(bert_model_name_eng)
# Model thai
bert_model_name_thai = 'airesearch/wangchanberta-base-att-spm-uncased'
model_thai = BERTClassifier(bert_model_name_thai, num_classes).to(device)
model_thai.load_state_dict(torch.load("bert_classifier.pth", map_location=device))
tokenizer_Autp_T = AutoTokenizer.from_pretrained(bert_model_name_thai)
# ฟังชั่นเกี่ยวกับพฤติกรรม แบบถ้าเจออันนี้ต้องไม่มีอันนี้
def remove_negations(L):
  for neg, pos in [('ไม่ออกกำลังกาย', 'ออกกำลังกาย'),
                   ('ไม่สูบบุหรี่', 'สูบบุหรี่'),
                   ('ไม่ดื่มเหล้า', 'ดื่มเหล้า'),('ออกกำลังกาย', 'ไม่ออกกำลังกาย'),
                   ('สูบบุหรี่', 'ไม่สูบบุหรี่'),
                   ('ดื่มเหล้า', 'ไม่ดื่มเหล้า')]:
    if neg in L and pos in L: # Only remove 'pos' if both 'neg' and 'pos' are present
      L.remove(pos)
import random
def process_data():
    time.sleep(5)  # จำลองการประมวลผลที่ใช้เวลา 5 วินาที
    # Read the CSV file into a pandas DataFrame
    df_facts = pd.read_csv('ncds_facts.csv')

    # Extract the 'ข้อเท็จจริง' column as a list
    facts = df_facts['ข้อเท็จจริง'].tolist()
    facts
    random.shuffle(facts) # สลับลำดับ facts
    return facts[:1] # แสดง 3 facts แรก
server = Flask(__name__,static_folder=os.path.join(os.getcwd(),'static'))


@server.route('/', methods=['GET'])
# ส่วนของหน้าที่ 1 ให้เลือกวิธีเอาข้อมูลเข้า แรกสุด
def index_A():
    ref = {'ref':[0],'refy':[0]}
    select = {'use_url':[0]}
    time_run_va = {'time_run':[0]}
    ref_pas = pd.DataFrame(ref)
    select_1 = pd.DataFrame(select)
    time_run= pd.DataFrame(time_run_va)
    ref_pas.to_csv('ref_pas.csv',index=False)
    select_1.to_csv('use_F_or_U.csv',index=False)
    time_run.to_csv('time_run.csv',index=False)
    return render_template('pagr_1_NEW.html')

@server.route('/get_facts')
def get_facts():
    facts = process_data()
    return jsonify({'facts': facts})

@server.route('/page1_2', methods=['GET','POST'])
# ส่วนของหน้าที่ 1 ให้เลือกวิธีเอาข้อมูลเข้า แต่ผ่านการเอาข้อมูลเข้ามาแล้ว
def index_AA():
    time_run = pd.read_csv('time_run.csv')
    time_run.iloc[0,0] = 0
    time_run.to_csv('time_run.csv',index=False)
    x00 = pd.read_csv('ref_pas.csv')
    x10 = x00['ref'][0]
    x30 = x00['refy'][0]
    return render_template('pagr_1_NEW1.html',x=x10,y=x30)

@server.route('/url_sc', methods=['GET','POST'])
# ส่วนของหน้าที่ 1 สำหรับ input url ใช้ประกอบกับ HTML
def index_B():
  x01 = pd.read_csv('ref_pas.csv')
  x11 = x01['ref'][0]
  x30 = x01['refy'][0]
  return render_template('Page1_pro1.html',x=x11,y=x30)

@server.route('/use_File', methods=['GET','POST'])
# ส่วนของหน้าที่ 1 สำหรับ input file ใช้ประกอบกับ HTML
def index_C():
  x02 = pd.read_csv('ref_pas.csv')
  x12 = x02['ref'][0]
  x30 = x02['refy'][0]
  return render_template('Page 2 use_file.html',x=x12,y=x30)
# ส่วนของหน้าที่ 1 สำหรับการดึงข้อมูลมาเก็บเป็นตาราง เเละให้เเสดงผลหน้าที่ 2 โดยให้เเสดงตารางข้อมูล,ค่าสถิติเบื้องต้น,word cloud,เเละการใส่อาการ ใช้ร่วมกับ HTML ด้วย
# ปล. ตัวกรอกโรคและอาการเพิ่มเป็น tag ต่างๆใช้html(เขียนเอง),jqurry(เขียนเอง) เเละ css(สำเร็จรูป)
@server.route('/process.py', methods=['POST'])
def process():
    import pandas as pd
    # ส่วนของการเก็บข้อมูลจาก advance option
    url = request.form['url']
    x_1 = request.form['x']
    a_1 = request.form['a']
    aoa_1 =request.form['aoa']
    r_1 = request.form['r']
    flie_name = 'Data_scraper_save.csv'
    # ส่วนของการเก็บข้อมูล ว่าเราดึงจากเว็ปไหน เช่น ดึงจากFacebook ก็จะเป็น www.Facebook.com
    url_chack = str(url).split('/')[2]
    data_soure_a = {'url':[url_chack]}
    data_soure_b = pd.DataFrame(data_soure_a)
    data_soure_b.to_csv('soure_url.csv',index=False)
    # ดึงข้อมูลจาก facebook
    if url_chack == 'www.facebook.com':
        # สกัดองค์ความรู้
        option = Options()
        option.add_argument("--disable-infobars")
        option.add_argument("start-maximized")
        option.add_argument("--disable-extensions")
        # Pass the argument 1 to allow and 2 to block
        # โดยปกติเวลาเราเข้าหน้าเว็ปที่มีการขอข้อมูลจะมีการเด้ง "popup" ให้เรากดยอมรับ(allow)หรือปิดกัน(block)
        option.add_experimental_option(
            "prefs", {"profile.default_content_setting_values.notifications": 1}
        )
        # โดยปกติเวลาเราเข้า facebook ผ่าน webdriver Chrome มันจะไม่มีการ login Facebook ให้เเละหากไม่login 
        # Facebook เองก็จะพยายามให้เรา login ส่งผลให้เราติดที่หน้าlogin
        # code ด้านล่างนี้ก็จะเป็นส่วนของการ auto_login ผ่านข้อมูล ชื่อเเละรหัสผ่านในไฟล์ facebook_credentials.txt
        with open('facebook_credentials.txt') as file:
            EMAIL = file.readline().split('"')[1]
            PASSWORD = file.readline().split('"')[1]
            
        browser = webdriver.Chrome(options=option)
        browser.get("http://facebook.com")
        browser.maximize_window()
        wait = WebDriverWait(browser, 30)
        email_field = wait.until(EC.visibility_of_element_located((By.NAME, 'email')))
        email_field.send_keys(EMAIL)
        pass_field = wait.until(EC.visibility_of_element_located((By.NAME, 'pass')))
        pass_field.send_keys(PASSWORD)
        pass_field.send_keys(Keys.RETURN)

        time.sleep(60)
        x=x_1
        a=int(x[162:164])
        k=x
        aoa=aoa_1
        kok={'a_a':'div/div/div','b_a':'div[2]/div/div','b_b':'div/div[2]/div','b_c':'div/div/div[2]','c_a':'div[3]/div/div','c_b':'div/div[3]/div','c_c':'div/div/div[3]'}
        LOL=[]
        time.sleep(20)
        browser.get(url)
        wait = WebDriverWait(browser, 120) # once logged in, free to open up any target page
        time.sleep(7)
        
        count_r = 0
        switch = True
        # เปิดความคิดเห็นเพิ่มเติม
        try: 
            while switch:
                count_r += 1
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                l = browser.find_element(By.XPATH,k)
                l.click()
                a+=int(a_1)
                k = x[:162]+str(a)+x[164:]
                time.sleep(7)
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(11)
        except:
            pass

         # เช็กว่าลงสุดไหม
        SCROLL_PAUSE_TIME = 4

        # Get scroll height
        last_height = browser.execute_script("return document.body.scrollHeight")
        while True:
            for i in range(5):
                # Scroll down to bottom
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load page
                time.sleep(SCROLL_PAUSE_TIME)
           
            # Calculate new scroll height and compare with last scroll height
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # เปิดเพิ่มเติม
        r=int(aoa[-81:-80])
        for i in range(2):
            for i in range(9999):
                r += int(r_1)
                for j in kok:
                    kra = aoa[:-81]+ str(r) + aoa[-80:-14] + kok[j]
                    LOL.append(kra)
            for g in range(1000):
                try:
                    l = browser.find_element(By.XPATH,LOL[g])
                    l.click()
                    time.sleep(1)
                except:
                    pass
            list_like_1 = []
            list_like_2 = []
            list_like_3 = []
            list_like_4 = []
            list_like_5 = []
            list_like_0 = []
            for i in range(9999):     
                          
                aoa_0 = f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i}]/div/div[1]/div[2]/div[2]/div[1]/div[1]/div/div/div/span/div/div/div'
                list_like_0.append(aoa_0)
                aoa_1 = f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i}]/div/div[1]/div[2]/div[2]/div[1]/div[1]/div/div/div[2]/span/div/div[3]/div'
                list_like_1.append(aoa_1)
                aoa_2 = f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i}]/div/div[1]/div/div[2]/div[1]/div[1]/div/div/div[2]/span/div/div[3]/div'
                list_like_2.append(aoa_2)
                aoa_3 = f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i}]/div/div[1]/div/div[2]/div[1]/div[1]/div/div/div/span/div/div[3]/div'
                list_like_2.append(aoa_3)
                aoa_4 = f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i}]/div/div[1]/div/div[2]/div[1]/div[1]/div/div/div/span/div/div/div'
                list_like_4.append(aoa_4)
                aoa_5 = f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i}]/div/div[1]/div[2]/div[1]/div[1]/div/div/div/span/div/div/div'
                list_like_5.append(aoa_5)
            for g in range(1000):
                try:
                    more_1 = browser.find_element(By.XPATH,list_like_1[g])
                    more_1.click()
                    time.sleep(1)
                except:
                    pass
            browser.execute_script("window.scrollTo(0,0);")
            for g in range(1000):
                try:
                    more_2 = browser.find_element(By.XPATH,list_like_2[g])
                    more_2.click()
                    time.sleep(1)
                except:
                    pass
            browser.execute_script("window.scrollTo(0,0);")
            for g in range(1000):
                try:
                    more_2 = browser.find_element(By.XPATH,list_like_3[g])
                    more_2.click()
                    time.sleep(1)
                except:
                    pass
            browser.execute_script("window.scrollTo(0,0);")
            for g in range(1000):
                try:
                    more_2 = browser.find_element(By.XPATH,list_like_4[g])
                    more_2.click()
                    time.sleep(1)
                except:
                    pass
            browser.execute_script("window.scrollTo(0,0);")
            for g in range(1000):
                try:
                    more_2 = browser.find_element(By.XPATH,list_like_5[g])
                    more_2.click()
                    time.sleep(1)
                except:
                    pass
        # เช็กว่าลงสุดไหมv.2
        last_height = browser.execute_script("return document.body.scrollHeight")
        while True:
            for i in range(3):
                # Scroll down to bottom
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load page
                time.sleep(5)
                # Calculate new scroll height and compare with last scroll height
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        #setup bs4 and data
        #ดึงข้อมูล
        comments=[]
        names=[]
        re_chat_all=[]
        like_kk = []
        count=[]
        html =  BeautifulSoup(browser.page_source, "html.parser")
        soup = BeautifulSoup(browser.page_source, 'lxml')
        dom = etree.HTML(str(soup))
        # ## comment
        re_chak = dom.xpath('/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[3]/div/div/div/div/div[2]/div/div/text()')
        test_chak = ''
        for item_chak in re_chak:
            test_chak = test_chak + item_chak
        re_chak1 = dom.xpath('/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div[4]/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[3]/div/div/div/div/div[2]/div/div/text()')
        test_chak1 = ''
        for item_chak in re_chak1:
            test_chak1 = test_chak1 + item_chak
        re_chak2 = dom.xpath('/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[2]/div/div/div/div[4]/div/div/div/div/div/div/div/div[1]/div[2]/div/div[1]/div/div/div/div/div/div[2]/div[1]/div/div/div/div/div/div/div/span/span/text()')
        test_chak2 = ''
        for item_chak in re_chak2:
            test_chak2 = test_chak2 + item_chak
        result = html.find_all('div',{"class":"xdj266r x11i5rnm xat24cr x1mh8g0r x1vvkbs"})
        # ## name
        name = html.find_all(["span"],{"class":"x193iq5w xeuugli x13faqbe x1vvkbs x1xmvt09 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x676frb x1nxh6w3 x1sibtaa x1s688f xzsf02u"})
        # ## re_chat
        for i in range(len(result)):
            x1=dom.xpath(f'/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div/div/div[2]/div/div/div[2]/div[2]/span/span/text()')
            x2 = dom.xpath(f'/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div/div/div[2]/div/div/div[2]/div[2]/span/span/text()')
            y1=dom.xpath(f'/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div/div/div[2]/div/div/div[2]/div[2]/span/span/div/div[4]/text()')
            y2=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[2]/div/div/div[2]/div[2]/span/span/text()')
            x3=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div[4]/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[2]/div/div/div[2]/div[2]/span/span/div/div[4]/span/span/text()')
            x_4=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[2]/div/div/div[2]/span/span/div/div[4]/span/span/text()')
            x_6=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[2]/div/div/div[2]/div[2]/span/span/text()')
            x_5=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[2]/div/div/div[2]/span/span/text()')
            x_7=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[2]/div/div/div[2]/div[2]/span/span/div/div[4]/span/span/text()')
                            
            z1=x1+y1+x2+y2+x3+x_4+x_5+x_6+x_7
            try:
                re_chat_all.append((re.findall(r'\b\d+\b',z1[0]))[0])
            except: 
                re_chat_all.append(0)
        # like
        for i in range(len(result)):
                             
            x2_1=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x2=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div[4]/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x3=dom.xpath(f'/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div/div/div[1]/div[2]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x4=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[2]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x5=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[2]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div/div[2]/div/div/div/span/div/div[1]/span/text()')
            x6=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x7=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div/div/div/div/span/div/div[1]/span/text()')
            x8=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[4]/div/div/div/div/div/div/div[1]/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div/div/div/div/div/div/span/div/div[1]/span/text()')
            x9=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x10=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x11=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            x12=dom.xpath(f'/html/body/div[1]/div/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div/div/div/div/div/div/div[2]/div/div/div[4]/div/div/div[2]/div[3]/div[{i+2}]/div/div[1]/div[2]/div[2]/div[2]/div/div/div/span/div/div[1]/span/text()')
            sum_like = x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x2_1+x12
            try:
                like_kk.append(sum_like[0])
            except:
                like_kk.append(0)

        # แปลง like จาก str ไปเป็น int
        like_count=[]
        for j in like_kk:
            try:
                split_v_like=j.split(' ')
                if split_v_like[1] == 'พัน':
                    ror=float(split_v_like[0])*1000
                    T_V_like=int(ror)
                elif split_v_like[1] == 'หมื่น':
                    ror=float(split_v_like[0])*100000
                    T_V_like=int(ror)
                elif split_v_like[1] == 'แสน':
                    ror=float(split_v_like[0])*1000000
                    T_V_like=int(ror)
                elif split_v_like[1] == 'ล้าน':
                    ror=float(split_v_like[0])*100000000
                    T_V_like=int(ror)
                like_count.append(T_V_like)
            except:
                like_count.append(int(j))
                
        #เริ่มเก็บข้อมูล
        FBcomments = []
        for item in result:
            comments.append(item.text)
        for item in comments:
            if item == test_chak:
                pass
            elif item == test_chak1:
                pass
            elif item == test_chak2:
                pass
            else:
                FBcomments.append(item)
        for item in name :
            names.append(item.text)
        data = pd.DataFrame()
        data_comment=pd.DataFrame(data={'comments':FBcomments})
        data_name=pd.DataFrame(data={'name':names})
        data_rechat=pd.DataFrame(data={'จำนวนการตอบกลับ':re_chat_all})
        data_like=pd.DataFrame(data={'ยอดไลค์':like_count})

        # ทำตาราง
        data = data_comment
        data = data.join(data_name).fillna('NaN_name')
        data = data.join(data_rechat).fillna(0)
        data = data.join(data_like).fillna(0)
        data = data.iloc[:,[1,0,3,2]]
        #data['url'] = url
        data['test'] = 101
        number_of_rows = len(data)
        number_of_columns = len(data.columns)
        data.to_csv(f'{flie_name}', index=False, encoding='utf-8-sig')
        browser.close()
        
    # reddit
    elif url_chack == 'www.reddit.com':
        import pandas as pd
        names=[]
        comments=[]
        data = pd.DataFrame()
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        # button = driver.find_element("CLASS",'text-tone-2 text-12 no-underline hover:underline px-xs py-xs flex ml-[3px] xs:ml-0 !bg-transparent !border-0')
        # button.click()
        try: 
            for i in range(99) :
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                wait = WebDriverWait(driver, 60)
                time.sleep(3)
                l = driver.find_element("xpath",'/html/body/shreddit-app/div/div[1]/div/main/div/faceplate-batch/shreddit-comment-tree/faceplate-partial/div[1]/faceplate-tracker/button')
                l.click()
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                wait = WebDriverWait(driver, 60)
                time.sleep(3)
        except:
            pass
        time.sleep(5)
        html =  BeautifulSoup(driver.page_source, "html.parser")
        result = html.find_all('div',{"class":"py-0 xs:mx-xs mx-2xs inline-block max-w-full"})
        name = html.find_all(['a',"span"],{"class":["truncate font-bold text-neutral-content-strong text-12 hover:underline","truncate font-bold text-neutral-content-strong text-12 hover:no-underline text-neutral-content-weak"]})
        for item in result:
            comments.append(item.text)
        for item in name :
            names.append(item.text)
        if len(comments)>len(names):
            differ = len(comments)-len(names)
            for i in range(differ):
                names.append(f'fillname_{i}')
        elif len(comments)<len(names):
            differ = len(names)-len(FBcomments)
            for i in range(differ):
                FBcomments.append('comments_miss')
        data['name'] = names
        data['comments'] = comments
        #data['url'] = url
        data['test'] = 101
        data=data.applymap(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
        data_red = data[data['comments'] != 'comments_miss']
        number_of_rows = len(data)
        number_of_columns = len(data.columns)
        data.to_csv(f'{flie_name}', index=False, encoding='utf-8-sig')
        driver.close() 
    filename = f'{flie_name}'
    x03 = pd.read_csv('ref_pas.csv')
    x13 = x03['ref'][0]
    x30 = x03['refy'][0]
    use_ur = pd.read_csv('use_F_or_U.csv')
    use_ur.iloc[0,0] = 1
    use_ur.to_csv('use_F_or_U.csv',index=False)
    return  render_template('Page 2 use_url.html',name = filename,x=x13,y=x30)

@server.route('/return-files/',methods=['POST','GET'])
def return_files_tut():
    data = pd.read_csv('Data_scraper_save.csv')
    data = data.drop(['test'], axis=1)
    #data = data.drop(['url'], axis=1)
    data.to_csv('Data_scraper.csv',index=False,encoding='utf-8-sig')
    # df = pd.read_csv('Data_scraper.csv', encoding='utf-8-sig')
    # resp = make_response(df.to_csv(index=False, encoding='utf-8-sig'))
    # resp.headers["Content-Disposition"] = "attachment; filename=Data_scraper.csv"
    # resp.headers["Content-Type"] = "text/csv"
    return send_file('Data_scraper.csv',
        mimetype='text/csv',
        download_name='Data_scraper.csv',
        as_attachment=True
    )

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server.secret_key = secrets.token_urlsafe(16)
@server.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')
 
        # Extracting uploaded file name
        data_filename =  secure_filename(f.filename)
 
        f.save(os.path.join(server.config['UPLOAD_FOLDER'],
                            data_filename))

        session['uploaded_data_file_path'] =os.path.join(server.config['UPLOAD_FOLDER'],data_filename)
        url_chack = request.form['language']
        data_soure_a = {'url':[url_chack]}
        data_soure_b = pd.DataFrame(data_soure_a)
        data_soure_b.to_csv('soure_url.csv',index=False)
        x04 = pd.read_csv('ref_pas.csv')
        x14 = x04['ref'][0]
        x30 = x04['refy'][0]
        use_ur = pd.read_csv('use_F_or_U.csv')
        use_ur.iloc[0,0] = 0
        use_ur.to_csv('use_F_or_U.csv',index=False)
        return render_template('Page_use_file_test.html',x=x14,y=x30)
    x05 = pd.read_csv('ref_pas.csv')
    x15 = x05['ref'][0]
    x30 = x05['refy'][0]
    return render_template("Page 2 use_file.html",x=x15,y=x30)#D:\new_pro\templates\Page 2 use_file_after.html

@server.route('/after_pepar', methods = ['POST','GET'])   
def success():   
    import pandas as pd
    use_ur_or_fi = pd.read_csv('use_F_or_U.csv')
    if use_ur_or_fi.iloc[0,0] == 1:
        data = pd.read_csv('Data_scraper_save.csv')   
        data = data.drop(['test'], axis=1)
        data.to_csv('Data_scraper_save.csv',index=False,encoding='utf-8-sig')
        data.to_csv('Data_prepo.csv',index=False,encoding='utf-8-sig')   
    elif use_ur_or_fi.iloc[0,0] == 0:
        data_file_path = session.get('uploaded_data_file_path', None)
        data= pd.read_csv(data_file_path,encoding='utf-8-sig')
        data.to_csv('Data_prepo.csv',index=False,encoding='utf-8-sig') 
    data = pd.read_csv('Data_prepo.csv',encoding='utf-8-sig')
    count_user = len(set(data['name']))
    count_comment = len(set(data['comments']))
    number_of_rows = len(data)
    number_of_columns = len(data.columns)
    sorue_sym_2 = pd.read_csv('soure_url.csv')
    sorue = sorue_sym_2['url'][0]
    if sorue == 'www.facebook.com':
        data_sy_na = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
        data_defind = data_sy_na[['name_cancarTH','Key_symptoms_TH','Values_symptoms_TH']]
        data_defind = data_defind.rename(columns={'name_cancarTH': 'name_cancarTH_se'})
        data_defind.to_csv('all_data_nameandsym.csv', index=False, encoding='utf-8-sig')
    elif sorue == 'www.reddit.com':
        data_sy_na = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
        data_defind = data_sy_na[['cancer_names_en','Key_symptoms_EN','Valuessymptoms_EN']]
        data_defind=data_defind.rename(columns={'cancer_names_en': 'cancer_names_en_se'})
        data_defind.to_csv('all_data_nameandsym.csv')
    sorue_sym_3 = pd.read_csv('soure_url.csv')
    sorue = sorue_sym_3['url'][0]
    if sorue == 'www.facebook.com':
        # ระบุโรค
        name_cancar_and_symptoms_pd = pd.read_csv('all_data_nameandsym.csv')
        name_cancar_list  = name_cancar_and_symptoms_pd['name_cancarTH_se'].tolist()
        name_cancar = [item for item in name_cancar_list if not(pd.isnull(item)) == True]
        # ระบุอาการ
        symptoms_pd = name_cancar_and_symptoms_pd[['Key_symptoms_TH','Values_symptoms_TH']].dropna()
        data_k_list = symptoms_pd['Key_symptoms_TH'].tolist()
        data_V_list = symptoms_pd['Values_symptoms_TH'].tolist()
        symptoms = {}
        for list_item in range(len(data_V_list)):
            split_t = ast.literal_eval(data_V_list[list_item])
            split_t = [n.strip() for n in split_t]
            symptoms[data_k_list[list_item]]=split_t

        #เเปลงหัวตาราง
        comment=data.groupby('name').sum().reset_index()

        # สร้าง set ข้อมูลภาษาไทย
        from pythainlp.corpus import thai_stopwords # เป็นคลัง Stop Words ของภาษาไทย
        words = set(thai_words())
        remove_name = ["โรคเบาหวาน","โรคมะเร็ง",'โรคหัวใจ',"เป็นโรค","โรคความดันโลหิตสูง"]
        stop_w = set(thai_stopwords())
        remove_name_sw = ["ไม่", "เร็ว"]
        for i in remove_name_sw:
            stop_w.remove(i)
        for i in remove_name:
            words.remove(i)
        name =['กระเพาะปัสสวะ','กระเพาะปัสสาวะ','เยื่อบุโพรงมดลูก','ปากมดลูก','เม็ดเลือดขาว','กระเพาะอาหาร','กระเพราะอาหาร','ต่อมไทรอยด์','ต่อมไทยรอยด์','ท่อน้ำดี','เต้นแรง',"เต้นเร็ว", "เต้นช้า", "หยุดเต้น"
               , 'เต้นรัว', 'หยุดเต้น', 'จี้ไฟฟ้า', 'ยาลดความดัน', 'ยาลดไขมัน','กินยา']
        for i in name:
            words.add(i)
        def remove_stopthai(tokens):
            final = [word for word in tokens if word not in stop_w]
            return final
        def remove_bark(tokens):
            final = [word for word in tokens if word not in [' ','(',')','@','#','-','...','.','=','+','..','1','2','3','4','5','6','7','8','9','0','*']]
            return final
        # สร้าง list เก็บตัว nlp เพิ่อนำไปวิเคราะห์โรค อาการ เเละเพศ
        custom_tokenizer = Tokenizer(words)
        list_token = []
        count = []

        for text in comment['comments']:  # Iterate directly over the comment column
            normalized_text = normalize(str(text))
            tokens = custom_tokenizer.word_tokenize(normalized_text)
            tokens.append('end')
            token_final = remove_bark(remove_stopthai(tokens))
            list_token.append(token_final)
            count.append(len(tokens) - 1)

        comment['จำนวนคำ'] = count
        comment['token'] = list_token
        # 2-gamr
        token_2gam = []
        for i in range(len(comment)):
            pop = []
            for tojo in range(len(comment['token'].iloc[i])):
                try:
                    pop.append(comment['token'].iloc[i][tojo]+comment['token'].iloc[i][tojo+1])
                except:
                    pass
            token_2gam.append(pop)
        # 3-gamr
        token_3gam = []
        for i in range(len(comment)):
            pop = []
            for tojo in range(len(comment['token'].iloc[i])):
                try:
                    pop.append(comment['token'].iloc[i][tojo]+comment['token'].iloc[i][tojo+1]+comment['token'].iloc[i][tojo+2])
                except:
                     pass
            token_3gam.append(pop)
        comment['token_2g'] = token_2gam
        comment['token_3g'] = token_3gam
        comment.to_csv('data_tokenizer.csv',encoding='utf-8-sig')
        time.sleep(8)
        #หาโรค
        #--------------------------------------------------------
        new_colcan = []
        for i in range(len(list_token)):#len(comment)
            list_cancer = []
            for k in range(len(list_token[i])):
                if (list_token[i][k] == "มะเร็ง")|(list_token[i][k] == "โรคมะเร็ง")|(list_token[i][k] == "โรค")|(list_token[i][k] == "หัวใจ"):
                    list_cancer.append(list_token[i][k]+list_token[i][k+1])
                elif (list_token[i][k] == "เบาหวาน")|(list_token[i][k] == "อ้วน")|(list_token[i][k] == "ตับเเข็ง"):
                    list_cancer.append("โรค"+list_token[i][k])
                elif (list_token[i][k] == "ความดัน"):
                    list_cancer.append("โรค"+list_token[i][k]+"โลหิต")
                elif (list_token[i][k] == "ลิ้น"):
                    list_cancer.append("ลิ้น"+list_token[i][k+1]+"รั่ว")
            unique_list = list(OrderedDict.fromkeys(list_cancer))
            #----------------------------------------------------------
            list_define_cancer = []
            for i in range(len(unique_list)):
                for j in range(len(name_cancar)):
                    if unique_list[i]==name_cancar[j]:
                        list_define_cancer.append(unique_list[i])
            #-----------------------------------------------------
            cancer_list_de2 =[]
            if len(list_define_cancer) > 0:
                if len(list_define_cancer) == 2:
                    cancer_list_de2.append('เล่ามากกว่า 2 โรค')
                elif len(list_define_cancer)==1:
                    cancer_list_de2.append(list_define_cancer[0])
            elif list_define_cancer==[]:
                cancer_list_de2.append('ไม่สามารถระบุได้/ไม่มั่นใจว่าเป็น')
            if len(cancer_list_de2)> 0 :
                new_colcan.append(cancer_list_de2[0])
            elif len(cancer_list_de2)== 0 :
                new_colcan.append('ไม่สามารถระบุได้/ไม่มั่นใจว่าเป็น')
        # สร้าง list เพศ 
        Genden = {'ชาย':['พ่อ','บิดา','พี่ชาย','น้องชาย','ลูกชาย','สามี','พัว','ผัว','ปู่','ตา','คุณปู่','คุณตา','คุณพ่อ',
                        'ปู่ทวด','ตาทวด','ลุง','อาหนู','คุณอา','คุณลุง','หลายชาย','ลูกเขย','เขย','พี่เขย','น้องเขย',
                        'พ่อตา','พ่อผม','พ่อหนู','พ่อพม','ชาย','หนุ่ม','ลช.','ผ่อ','ชย.','น้าชาย','ผ่อตา','หน.']
                ,'หญิง':['แม่','เเม่','คุณแม่','มารดา','พี่สาว','น้องสาว','ลูกสาว','ภรรยา','เมีย','ย่า','ยาย','คุณย่า',
                        'คุณยาย','คุณเเม่','ย่าทวด','ยายทวด','ป้า','น้า','คุณป้า','คุณน้า','หลายสาว','ลูกสะใถ้',
                        'ลูกสะใภ้','สะใภ้','พี่สะใภ้','น้องสะใภ้','เเม่ผม','เเม่หนู','เเม่พม','แม่ผม','แม่หนู','แม่พม','สาว','หญิง','ก้อน','คลำ']}
        # หาเพศ โดยใช้nlp
        new_colgenden=[]
        list_genden=[]
        for i in range(len(list_token)):
            for j in range(len(Genden['หญิง'])):
                for k in range(len(list_token[i])):
                    if (list_token[i][k] == Genden['ชาย'][j]):
                        list_genden.append('เพศชาย')
                    elif(list_token[i][k] == Genden['หญิง'][j]):
                        list_genden.append('เพศหญิง')
            genden_list =[]
            genden_list = list(OrderedDict.fromkeys(list_genden)) # ลบคำซ้ำ
            #-------------------------------------------------------------------
            list_define_genden = []
            if len(genden_list) > 0 :
                if len(genden_list) == 2:
                    list_define_genden.append('เล่าทั้งสองเพศ')
                elif len(genden_list)==1:
                    list_define_genden.append(genden_list[0])
            elif len(genden_list)==0:
                list_define_genden.append('ไม่สามารถระบุได้')
            genden_list_de =[]
            genden_list_de = list(OrderedDict.fromkeys(list_define_genden))
            new_colgenden.append(genden_list_de[0])

        # หาอาการโดย nlp
        symptoms_colcan = []
        for i in range(len(list_token)):
            list_symptoms= []
            for j in range(len(list_token[i])):
                for k in symptoms:
                    for l in range(len(symptoms[k])):
                        if list_token[i][j] == symptoms[k][l]:
                            list_symptoms.append(symptoms[k][0])
            unique_list_symptoms = list(OrderedDict.fromkeys(list_symptoms))
            if len(unique_list_symptoms) > 0:
                symptoms_colcan.append(unique_list_symptoms)
            else :
                symptoms_colcan.append(['ไม่มีการระบุอาการ'])

        #สร้างฟังชั่นหาเพศกับใครเป็นคนพูด
        def detect_person(comment):
        # คำที่ใช้ตรวจสอบว่ามีใครเป็นคนอยู่ในความคิดเห็น
            other = ['พ่อ','บิดา','พี่ชาย','น้องชาย','ลูกชาย','สามี','พัว','ผัว','ปู่','ตา','คุณปู่','คุณตา','คุณพ่อ',
                    'ปู่ทวด','ตาทวด','ลุง','อาหนู','คุณอา','คุณลุง','หลายชาย','ลูกเขย','เขย','พี่เขย','น้องเขย',
                    'พ่อตา','พ่อผม','พ่อหนู','พ่อพม','ชาย','หนุ่ม','แม่','เเม่','คุณแม่','มารดา','พี่สาว','น้องสาว',
                    'ลูกสาว','ภรรยา','เมีย','ย่า','ยาย','คุณย่า','คุณยาย','คุณเเม่','ย่าทวด','ยายทวด','ป้า',
                    'น้า','คุณป้า','คุณน้า','หลายสาว','ลูกสะใถ้','ลูกสะใภ้','สะใภ้','พี่สะใภ้','น้องสะใภ้','เเม่ผม','เเม่หนู'
                    ,'เเม่พม','แม่ผม','แม่หนู','แม่พม','สาว','หญิง']
            myself = ['ผมเป็น','ผมเอง','กระผม','พมเป็น','พมเอง','กระพม','หนูเอง','หนู','ดิฉัน','ตัวเอง','ก้อน','คลำ']
            # ตรวจสอบคำในความคิดเห็น
            for keyword in  other:
                if keyword in comment and "อาการ" not in comment :
                    return 'เล่าประสบการณ์คนอื่น'
            for keyword in  myself:
                if keyword in comment:
                    return 'เล่าประสบการณ์ตัวเอง'
            # หากไม่พบคำที่บ่งบอกถึงคน
            return 'ไม่สามารถระบุได้'
        def detect_gender_other(comment):
            # คำที่บ่งบอกถึงเพศชาย
            male_keywords = ['พ่อ','บิดา','พี่ชาย','น้องชาย','ลูกชาย','สามี','พัว','ผัว','ปู่','ตา','คุณปู่','คุณตา','คุณพ่อ',
                            'ปู่ทวด','ตาทวด','ลุง','อาหนู','คุณอา','คุณลุง','หลายชาย','ลูกเขย','เขย','พี่เขย','น้องเขย',
                            'พ่อตา','พ่อผม','พ่อหนู','พ่อพม','ชาย','หนุ่ม']
            # คำที่บ่งบอกถึงเพศหญิง
            female_keywords = ['แม่','เเม่','คุณแม่','มารดา','พี่สาว','น้องสาว','ลูกสาว','ภรรยา','เมีย','ย่า','ยาย','คุณย่า','คุณยาย','คุณเเม่','ย่าทวด','ยายทวด','ป้า',
                            'น้า','คุณป้า','คุณน้า','หลายสาว','ลูกสะใถ้','ลูกสะใภ้','สะใภ้','พี่สะใภ้','น้องสะใภ้','เเม่ผม','เเม่หนู','เเม่พม','แม่ผม','แม่หนู','แม่พม','สาว','หญิง']
            # ตรวจสอบคำในความคิดเห็น
            for keyword in male_keywords:
                if keyword in comment and "อาการ" not in comment :
                    return "เพศชาย"
            for keyword in female_keywords:
                if keyword in comment:
                    return "เพศหญิง"
            # หากไม่พบคำที่บ่งบอกถึงเพศ
            return "ไม่สามารถระบุได้"
        def detect_gender_self(comment):
            # คำที่บ่งบอกถึงเพศชาย
            male_keywords = ['ผมเป็น','ครับ','ผมเอง','คับ','กระผม','พมเป็น','พมเอง','กระพม']
            # คำที่บ่งบอกถึงเพศหญิง
            female_keywords = ['ค่ะ','คะ','หนูเอง','หนูเป็น','ดิฉัน','ก้อน','คลำ']
            # ตรวจสอบคำในความคิดเห็น
            for keyword in male_keywords:
                if keyword in comment:
                    return "เพศชาย"
            for keyword in female_keywords:
                if keyword in comment:
                    return "เพศหญิง"
            # หากไม่พบคำที่บ่งบอกถึงเพศ
            return "ไม่สามารถระบุได้"
        #เเบ่งเพศเเละใครเล่าโดยใช้ python
        k1=[]
        k2=[]
        for i in comment['comments']:
            k1.append(detect_person(str(i)))
            if detect_person(str(i)) == 'เล่าประสบการณ์คนอื่น':
                k2.append(detect_gender_other(str(i)))
            elif detect_person(str(i)) == 'เล่าประสบการณ์ตัวเอง':
                k2.append(detect_gender_self(str(i)))
            elif detect_person(str(i)) == 'ไม่สามารถระบุได้':
                k2.append(detect_gender_self(str(i)))
        # #หาพฤติกรรม
        behaviour_value ={'นอนไม่เป็นเวลา':['นอนดึก','นอนไม่เป็นเวลา',"อดนอน"],
                              'กินอาหารไม่ถูกสุขภาพ':['กินของทอด','กินของมัน','ไม่กินผัก','เค็มจัด','กินเค็ม','หวานจัด','กินหวาน','ไม่คุมอาหาร','อาหารทอด','ทอดอาหาร'
                                ,'กินของหมักดอง','กินของหมักดอง','กินของปิ้ง','กินของย่าง','ปิ้งย่าง'],
                                'ดื่มน้ำหวาน':['กินน้ำอัดลม', 'น้ำหวาน'],
                                'ไม่ออกกำลังกาย':['ไม่ออกกำลังกาย','ไม่เล่นกีฬา','ไม่ชอบออกกำลังกาย','กำลังกายไม่ค่อย','กำลังกายไม่ออก','ว่ายน้ำ', 'วิ่ง', 'ปั่นจักรยาน', 'กระโดดเชือก'],
                                'นอนตรงเวลา':['ตอนแต่วัน','นอนตรงเวลา'],
                                'กินอาหารสุขภาพ':['ไม่กินของ','กินผัก','ไม่กินเค็ม','ไม่กินหวาน','คุมอาหาร','ดื่มน้ำ','ลดหวาน','ลดเค็ม','คุมการกิน'],
                                'ออกกำลังกาย':['ออกกำลังกาย','เล่นกีฬา','ชอบออกกำลังกาย'],
                                'ไม่สูบบุหรี่':['ไม่สูบ','ไม่สูบบุหรี่','ไม่บุหรี่','เลิกสูบ','งดบุหรี่'],
                                'สูบบุหรี่':['ชอบสูบ','สูบบุหรี่','บุหรี่'],
                                'ไม่ดื่มเหล้า':['ไม่ดื่ม','ไม่ดื่มเหล้า','ไม่เหล้า','งดเหล้า'],
                                'ดื่มเหล้า':['ดื่มเหล้า','กินเหล้า','ดื่มแอลกอฮอล์'],
                                'อยู่ใกล้บุหรี่':['ใกล้คนสูบ','ครอบครัวสูบ','บุหรี่มือสอง'],
                                'อยู่ใกล้แหล่งมลผิษ':['ควัน','มลผิษ','ฝุ่นพง','ฝุ่น','ก่อสร้าง'],
                                "เครียด":["เครียด","อาการเครียด"],
                                "ทำงานหนัก":["ทำงานหนัก"]}
        behaviour_key = ['นอนไม่เป็นเวลา','กินอาหารไม่ถูกสุขภาพ','ดื่มน้ำหวาน','ไม่ออกกำลังกาย','นอนตรงเวลา','กินอาหารสุขภาพ','ออกกำลังกาย','ไม่สูบบุหรี่','สูบบุหรี่','ไม่ดื่มเหล้า','ดื่มเหล้า'
        ,'อยู่ใกล้บุหรี่','อยู่ใกล้แหล่งมลผิษ',"เครียด"]
        behaviour_extion = []
        for i in range(len(comment)):
            behaviour_list=[]
            for j in behaviour_key:
                for lol in behaviour_value[j]:
                # Use .iloc to access rows by numerical index
                    if lol in comment['token'].iloc[i]:
                        behaviour_list.append(j)
                    else:
                        behaviour_list.append('ไม่สามารถระบุได้')
            for j in behaviour_key:
                for lol in behaviour_value[j]:
                # Use .iloc to access rows by numerical index
                    if lol in comment['token_2g'].iloc[i]:
                        behaviour_list.append(j)
                    else:
                        behaviour_list.append('ไม่สามารถระบุได้')
            for j in behaviour_key:
                for lol in behaviour_value[j]:
                # Use .iloc to access rows by numerical index
                    if lol in comment['token_3g'].iloc[i]:
                        behaviour_list.append(j)
                    else:
                        behaviour_list.append('ไม่สามารถระบุได้')
            behaviour_list =list(set(behaviour_list))
            remove_negations(behaviour_list)
            if len(behaviour_list) >= 2:
                behaviour_list.remove('ไม่สามารถระบุได้')
            behaviour_extion.append(behaviour_list)
        # วิธีการรักษา
        Treatment_value ={'Surgery':['ผ่าตัด','Surgery','ผ่า','operation', 'incision', 'surgery','บายพาส','สวนหัวใจ','ลดขนาดกระเพาะอาหาร','ตัดกระเพาะอาหาร'],
                    'Chemotherapy':['chemotherapy','chemo','คีโม','เคมี','herceptin','บำบัดรักษา'],
                    'Radiation Therapy':['radiation','ฉายแสง','เเร่','ฉายเเสง'],
                    'Targeted Therapy':['targeted','เป้า'],
                    "Diet control": ["ลดแป้ง","ลดน้ำตาล","ลดไขมัน", "กินผัก" ,"กินผลไม้" ,"กินธัญพืช", "เพิ่มใยอาหาร","ลดคอเลสเตอรอล","ลดปริมาณอาหาร","ลดปริมาณ"
                                ,"ปรับเปลี่ยนพฤติกรรมการกิน", "เปลี่ยนการกิน","ลดโซเดียม", "ลดโพแทสเซียม",
                                "reducestarch", "reducesugar", "reducefat", "eatvegetables", "eatfruits", "eatgrains", "increasedietary", "reducecholesterol", "Reducefood", "reduceintake"
                                , "changeeat",  "reducesodium", "reducepotassium"],
                    "Drug use": ["อินซูลิน","ยาลดน้ำตาล","ยาลด","ยาลดความดัน", "ยาลดไขมัน","ยาลดความอยากอาหาร","กินยา",
                                 "insulin", "drugs", "diet drugs", "lowering", "suppressants", "medicine"],
                    "Self-care": ["ตรวจระดับน้ำตาล", "ดูแลเท้า","ลดเครียด","ลดความเครียด","พักผ่อนให้เพียงพอ","พักผ่อน",
                                  "bloodsugar", "care", "reducestress", "rest", "relax"],
                    }
        Treatment_key = ['Surgery','Chemotherapy','Radiation Therapy','Targeted Therapy','Diet control',"Drug use","Self-care"]
        Treatment_extion = []
        for i in range(len(comment)):
          Treatment_list=[]
          for j in Treatment_key:
            for lol in Treatment_value[j]:
              # Use .iloc to access rows by numerical index
                if lol in comment['token'].iloc[i]:
                    Treatment_list.append(j)
                elif lol in comment['token_2g'].iloc[i]:
                    Treatment_list.append(j)
                elif lol in comment['token_3g'].iloc[i]:
                    Treatment_list.append(j)
                else:
                    Treatment_list.append('ไม่สามารถระบุได้')
          Treatment_list =list(set(Treatment_list))
          if len(Treatment_list) >= 2:
            Treatment_list.remove('ไม่สามารถระบุได้')
          Treatment_extion.append(Treatment_list)
        # แยกความคิดเห็น
        # text_use = comment['comments'].tolist()
        # labels_use = []
        # for i in text_use:
        #     text_to_predict = i # Replace with your input
        #     predicted_sentiment = predict_sentiment(text_to_predict, model, tokenizer_Autp, device)
        #     labels_use.append(predicted_sentiment)
        Data_pre_and_clane = comment
        Data_pre_and_clane['defind_cancer_with_nlp'] = new_colcan
        # Data_pre_and_clane['use_ful'] = 'non'
        Data_pre_and_clane['defind_exp_with_python'] = k1
        Data_pre_and_clane['defind_Genden_with_nlp'] = new_colgenden
        Data_pre_and_clane['defind_Genden_with_python'] = k2
        Data_pre_and_clane['beha'] = behaviour_extion
        Data_pre_and_clane['Treatment'] = Treatment_extion
        comment_label = []
        for text in Data_pre_and_clane['comments']:
            predicted_sentiment = predict_sentiment_thai(text,model_thai=model_thai, tokenizer_Autp_T=tokenizer_Autp_T, device=device)
            comment_label.append(predicted_sentiment)
        Data_pre_and_clane['label'] = comment_label
        Data_pre_and_clane['symptoms_colcan_en']=symptoms_colcan
        label_symptoms=Data_pre_and_clane['symptoms_colcan_en'].str.join(sep='*').str.get_dummies(sep='*')
        Data_pre_and_clane=Data_pre_and_clane.join(label_symptoms)
        Data_pre_and_clane.to_csv('data_pre.csv', index=False, encoding='utf-8-sig')
        data_show = Data_pre_and_clane.iloc[:,:5]
    elif sorue == 'www.reddit.com': # ยังไม่เพิ่ม พฤติกรรม วิธีการรักษา ลักษณะความคิดเห็น
        import pandas as pd            
        # ระบุโรค
        name_cancar_and_symptoms_pd = pd.read_csv('all_data_nameandsym.csv')
        name_cancar_list  = name_cancar_and_symptoms_pd['cancer_names_en_se'].tolist()
        cancer_names_en = [item for item in name_cancar_list if not(pd.isnull(item)) == True]
        # ระบุอาการ
        symptoms_pd = name_cancar_and_symptoms_pd[['Key_symptoms_EN','Valuessymptoms_EN']].dropna()
        data_k_list = symptoms_pd['Key_symptoms_EN'].tolist()
        data_V_list = symptoms_pd['Valuessymptoms_EN'].tolist()
        # ระบุโรคเเละอาการในรูป lower
        cancer_symptoms_en = {}
        for list_item in range(len(data_V_list)):
            split_t = ast.literal_eval(data_V_list[list_item])
            split_t = [n.strip() for n in split_t]
            cancer_symptoms_en[data_k_list[list_item]]=split_t
        data_raddit=data.groupby('name').sum().reset_index()
        def token(data):
            tokens = nltk.word_tokenize(data)
            tokens.insert(0, 'start')
            tokens.append('end')
            return tokens
        def remove_(tokens):
        # โค้ดส่วนอื่น ๆ ที่ใช้ stopwords
            stop_words = set(stopwords.words('english'))
            final = [word.lower()
                    for word in tokens if word not in stop_words]
            return final
        # Lemmatizing
        def lemma(final):
            str1 = ' '.join(final)
            s = TextBlob(str1)
            lemmatized_sentence = " ".join([w.lemmatize() for w in s.words])
            return final
        data_com = data_raddit['comments'].to_list()
        list_token_red = []
        for item_comment_red in range(len(data_com)):
            data_r_token = token(data_com[item_comment_red].translate(str.maketrans('', '', string.punctuation)))
            data_r_token = remove_(data_r_token)
            data_r_token = lemma(data_r_token)
            list_token_red.append(data_r_token)
            count_1=[]
            for item in list_token_red:
                count_1.append(len(item))
        data_raddit['จำนวนคำ'] = count_1
        data_raddit['ยอดไลค์'] = 0
        data_raddit['จำนวนการตอบกลับ'] = 0
        data_raddit['token'] = list_token_red
        # 2-gamr
        token_2gam = []
        for i in range(len(data_raddit)):
            pop = []
            for tojo in range(len(data_raddit['token'].iloc[i])):
                try:
                    pop.append(data_raddit['token'].iloc[i][tojo]+" "+data_raddit['token'].iloc[i][tojo+1])
                except:
                    pass
            token_2gam.append(pop)
        # 3-gamr
        token_3gam = []
        for i in range(len(data_raddit)):
            pop = []
            for tojo in range(len(data_raddit['token'].iloc[i])):
                try:
                    pop.append(data_raddit['token'].iloc[i][tojo]+" "+data_raddit['token'].iloc[i][tojo+1]+" "+data_raddit['token'].iloc[i][tojo+2])
                except:
                     pass
            token_3gam.append(pop)
        data_raddit['token_2g'] = token_2gam
        data_raddit['token_3g'] = token_3gam
        data_raddit.to_csv('data_tokenizer.csv',encoding='utf-8-sig')
        time.sleep(8)
        column_cancer_nlp_rad=[]
        for list_token_i in range(len(list_token_red)):#len(comment)
            list_cancer_en = []
            for list_token_k in range(len(list_token_red[list_token_i])):
                if (list_token_red[list_token_i][list_token_k] == 'cancer' or list_token_red[list_token_i][list_token_k] == 'disease'):
                    list_cancer_en.append(list_token_red[list_token_i][list_token_k-1]+' '+list_token_red[list_token_i][list_token_k])
                else:
                    for j in range(len(cancer_names_en)):
                        if (list_token_red[list_token_i][list_token_k] == cancer_names_en[j]):
                            list_cancer_en.append(list_token_red[list_token_i][list_token_k])
            unique_list_en = list(OrderedDict.fromkeys(list_cancer_en))
        #----------------------------------------------------------------------------
            list_define_cancer_en = []
            for i in range(len(unique_list_en)):
                for j in range(len(cancer_names_en)):
                    if unique_list_en[i]==cancer_names_en[j]:
                        list_define_cancer_en.append(unique_list_en[i])
        #----------------------------------------------------------------------------
            cancer_list_de_red =[]
            if len(list_define_cancer_en) > 0:
                if len(list_define_cancer_en) == 2:
                    cancer_list_de_red.append('Tell more than 2 diseases.')
                elif len(list_define_cancer_en)==1:
                    cancer_list_de_red.append(list_define_cancer_en[0])
            elif list_define_cancer_en==[]:
                cancer_list_de_red.append('Unable to identify / not sure if it is')
            if len(cancer_list_de_red)> 0 :
                column_cancer_nlp_rad.append(cancer_list_de_red[0])
            elif len(cancer_list_de_red)== 0 :
                column_cancer_nlp_rad.append('Unable to identify / not sure if it is')
        #-----------------------------------
        all_reddit_data = data_raddit
        all_reddit_data['defind_cancer_with_nlp'] = column_cancer_nlp_rad
        #-----------------------------------
        # data_value_red =[]
        # for i in range(len(list_token_red)):
        #     if len(list_token_red[i]) >= 50 :
        #         data_value_red.append('maybe_useful')
        #     else:
        #         data_value_red.append('Not useful or not giving too much information')
        new_colgenden_en=[]
        list_genden_en=[]
        #เเบ่งเพศ
        Genden_en = {'Male':['dad', "father","stepfather","grandfather","great-grandfather",
        "husband","boyfriend","fiancé","son","stepson","grandson","great-grandson","brother",
        "half-brother","stepbrother","uncle","great-uncle","nephew","great-nephew",
        "father-in-law","stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law",
        "grandson-in-law","brother-in-law","half-brother-in-law","stepbrother-in-law","uncle-in-law",
        "great-uncle-in-law","nephew-in-law","great-nephew-in-law","godfather","stepfather",
        "godson","foster father","foster son","stepson","stepbrother","half-brother","great-uncle",
        "great-nephew","first cousin","second cousin","third cousin","father-in-law",
        "stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law","grandson-in-law",
        "brother-in-law","half-brother-in-law","stepbrother-in-law","uncle-in-law","great-uncle-in-law",
        "nephew-in-law","great-nephew-in-law"'he','him','his','boy'],
        'Female':["mother",'mom',"stepmother","grandmother","great-grandmother","wife",
        "girlfriend","fiancée","daughter","stepdaughter","granddaughter","great-granddaughter",
        "sister","half-sister","stepsister","aunt","great-aunt","niece","great-niece",
        "mother-in-law","stepmother-in-law","grandmother-in-law","daughter-in-law",
        "stepdaughter-in-law","granddaughter-in-law","sister-in-law",
        "half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law",
        "niece-in-law","great-niece-in-law","cousin-in-law","godmother","stepmother",
        "goddaughter","foster mother","foster daughter","stepdaughter","stepsister",
        "half-sister","great-aunt","great-niece","mother-in-law","stepmother-in-law","grandmother-in-law",
        "daughter-in-law","stepdaughter-in-law","granddaughter-in-law","sister-in-law",
        "half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law",
        "niece-in-law","great-niece-in-law",'she','her','hers','mommy','girl']}
        for i in range(len(list_token_red)):
            for j in range(len(Genden_en['Female'])):
                for k in range(len(list_token_red[i])):
                    if (list_token_red[i][k] == Genden_en['Male'][j]):
                        list_genden_en.append('เพศชาย')
                    elif(list_token_red[i][k] == Genden_en['Female'][j]):
                        list_genden_en.append('เพศหญิง')
            genden_list_en =[]
            genden_list_en = list(OrderedDict.fromkeys(list_genden_en)) # ลบคำซ้ำ
            #-------------------------------------------------------------------
            list_define_genden_en = []
            if len(genden_list_en) > 0 :
                if len(genden_list_en) == 2:
                    list_define_genden_en.append('เล่าทั้งสองเพศ')
                elif len(genden_list_en)==1:
                    list_define_genden_en.append(genden_list_en[0])
            elif len(genden_list_en)==0:
                list_define_genden_en.append('ไม่สามารถระบุได้')
            genden_list_de_en =[]
            genden_list_de_en = list(OrderedDict.fromkeys(list_define_genden_en))
            new_colgenden_en.append(genden_list_de_en[0])
        def detect_person_en(comment):
            # คำที่ใช้ตรวจสอบว่ามีใครเป็นคนอยู่ในความคิดเห็น
            other = ['you','your','we','they','he','she','him','her','it']
            myself = ['my','myself']
            for keyword in  myself:
                if keyword in comment:
                    return "เล่าประสบการณ์ตัวเอง"
            for keyword in  other:
                if keyword in comment:
                    return "เล่าประสบการณ์คนอื่น"
            # หากไม่พบคำที่บ่งบอกถึงคน
            return "ไม่สามารถระบุได้"
        # #หาพฤติกรรม
        behaviour_value = {
            'Irregular Sleep': ['staying up late', 'inconsistent sleep schedule', 'not getting enough sleep'],
            'Unhealthy Diet': ['eating a lot of fried food', 'eating a lot of fatty food', 'not eating vegetables', 'eating very salty food', 'eating very sweet food', 'not watching what I eat', 'fried foods', 'deep-fried foods', 'pickled foods', 'fermented foods', 'grilled foods', 'roasted foods', 'grilling', 'barbecuing'],
            'Sugary Drinks': ['drinking soda', 'sweet drinks'],
            'Lack of Exercise': ['not exercising', 'not playing sports', 'not liking exercise', 'not exercising much', 'having trouble exercising', 'swimming', 'running', 'biking', 'jumping rope'],
            'Regular Sleep': ['waking up early', 'going to bed on time'],
            'Healthy Diet': ['not eating...', 'eating vegetables', 'not eating salty food', 'not eating sweets', 'watching what I eat', 'drinking water', 'cutting back on sugar', 'cutting back on salt', 'diet'],
            'Exercise': ['exercising', 'playing sports', 'liking exercise'],
            'Not Smoking': ['not smoking', 'not smoking cigarettes', 'Not...', 'quitting smoking', 'staying away from cigarettes'],
            'Smoking': ['enjoying smoking', 'smoking cigarettes'],
            'Not Drinking': ['not drinking', 'not drinking alcohol', 'non alcohol', 'staying away from alcohol'],
            'Drinking': ['drinking alcohol', 'drinking alcohol', 'drinking alcohol'],
            'Near Smokers': ['around people who smoke', 'family members smoke', 'secondhand smoke'],
            'Near Pollution': ['smoke', 'pollution', 'dust', 'dust', 'construction'],
            'Stress': ['stress', 'stress symptoms'],
            'Working Hard': ['working hard']
            }
        behaviour_key = ['Irregular Sleep', 'Unhealthy Diet', 'Sugary Drinks', 'Lack of Exercise', 'Regular Sleep', 'Healthy Diet', 'Exercise', 'Not Smoking', 'Smoking', 'Not Drinking', 'Drinking', 'Near Smokers', 'Near Pollution', 'Stress']
        behaviour_extion = []
        for i in range(len(data_raddit)):
            behaviour_list=[]
            for j in behaviour_key:
                for lol in behaviour_value[j]:
                # Use .iloc to access rows by numerical index
                    if lol in data_raddit['token'].iloc[i]:
                        behaviour_list.append(j)
                    else:
                        behaviour_list.append('ไม่สามารถระบุได้')
            for j in behaviour_key:
                for lol in behaviour_value[j]:
                # Use .iloc to access rows by numerical index
                    if lol in data_raddit['token_2g'].iloc[i]:
                        behaviour_list.append(j)
                    else:
                        behaviour_list.append('ไม่สามารถระบุได้')
            for j in behaviour_key:
                for lol in behaviour_value[j]:
                # Use .iloc to access rows by numerical index
                    if lol in data_raddit['token_3g'].iloc[i]:
                        behaviour_list.append(j)
                    else:
                        behaviour_list.append('ไม่สามารถระบุได้')
            behaviour_list =list(set(behaviour_list))
            remove_negations(behaviour_list)
            if len(behaviour_list) >= 2:
                behaviour_list.remove('ไม่สามารถระบุได้')
            behaviour_extion.append(behaviour_list)
        # วิธีการรักษา
        Treatment_value ={'Surgery':['ผ่าตัด','Surgery','ผ่า','operation', 'incision', 'surgery','บายพาส','สวนหัวใจ','ลดขนาดกระเพาะอาหาร','ตัดกระเพาะอาหาร'],
                    'Chemotherapy':['chemotherapy','chemo','คีโม','เคมี','herceptin','บำบัดรักษา'],
                    'Radiation Therapy':['radiation','ฉายแสง','เเร่','ฉายเเสง'],
                    'Targeted Therapy':['targeted','เป้า'],
                    "Diet control": ["ลดแป้ง","ลดน้ำตาล","ลดไขมัน", "กินผัก" ,"กินผลไม้" ,"กินธัญพืช", "เพิ่มใยอาหาร","ลดคอเลสเตอรอล","ลดปริมาณอาหาร","ลดปริมาณ"
                                ,"ปรับเปลี่ยนพฤติกรรมการกิน", "เปลี่ยนการกิน","ลดโซเดียม", "ลดโพแทสเซียม",
                                "reducestarch", "reducesugar", "reducefat", "eatvegetables", "eatfruits", "eatgrains", "increasedietary", "reducecholesterol", "reducefood", "reduceintake"
                                , "changeeat",  "reducesodium", "reducepotassium",'diet'],
                    "Drug use": ["อินซูลิน","ยาลดน้ำตาล","ยาลด","ยาลดความดัน", "ยาลดไขมัน","ยาลดความอยากอาหาร","กินยา",
                                 "insulin", "drugs", "diet drugs", "lowering", "suppressants", "medicine"],
                    "Self-care": ["ตรวจระดับน้ำตาล", "ดูแลเท้า","ลดเครียด","ลดความเครียด","พักผ่อนให้เพียงพอ","พักผ่อน",
                                  "bloodsugar", "care", "reducestress", "rest", "relax"],
                    }
        Treatment_key = ['Surgery','Chemotherapy','Radiation Therapy','Targeted Therapy','Diet control',"Drug use","Self-care"]
        Treatment_extion = []
        for i in range(len(data_raddit)):
          Treatment_list=[]
          for j in Treatment_key:
            for lol in Treatment_value[j]:
              # Use .iloc to access rows by numerical index
                if lol in data_raddit['token'].iloc[i]:
                    Treatment_list.append(j)
                elif lol in data_raddit['token_2g'].iloc[i]:
                    Treatment_list.append(j)
                elif lol in data_raddit['token_3g'].iloc[i]:
                    Treatment_list.append(j)
                else:
                    Treatment_list.append('ไม่สามารถระบุได้')
          Treatment_list =list(set(Treatment_list))
          if len(Treatment_list) >= 2:
            Treatment_list.remove('ไม่สามารถระบุได้')
          Treatment_extion.append(Treatment_list)
        def detect_gender_other_en(comment):
            # คำที่บ่งบอกถึงเพศชาย
            male_keywords = ['he','him','dad', "father","stepfather","grandfather","great-grandfather","husband",
            "boyfriend","fiancé","son","stepson","grandson","great-grandson","brother","half-brother","stepbrother","uncle","great-uncle","nephew",
            "great-nephew","father-in-law","stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law","grandson-in-law","brother-in-law",
            "half-brother-in-law","stepbrother-in-law","uncle-in-law","great-uncle-in-law","nephew-in-law","great-nephew-in-law","godfather","stepfather",
            "godson","foster father","foster son","stepson","stepbrother","half-brother","great-uncle","great-nephew","first cousin","second cousin",
            "third cousin","father-in-law","stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law","grandson-in-law","brother-in-law",
            "half-brother-in-law","stepbrother-in-law","uncle-in-law","great-uncle-in-law","nephew-in-law",
            "great-nephew-in-law",'boy']
            # คำที่บ่งบอกถึงเพศหญิง
            female_keywords = ["mother",'mom',"stepmother",'girl',
            "grandmother","great-grandmother","wife","girlfriend","fiancée","daughter","stepdaughter",
            "granddaughter","great-granddaughter","sister","half-sister","stepsister","aunt","great-aunt","niece",
            "great-niece","mother-in-law","stepmother-in-law","grandmother-in-law","daughter-in-law","stepdaughter-in-law",
            "granddaughter-in-law","sister-in-law","half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law",
            "niece-in-law","great-niece-in-law","cousin-in-law","godmother","stepmother","goddaughter","foster mother","foster daughter","stepdaughter","stepsister",
            "half-sister","great-aunt","great-niece","mother-in-law","stepmother-in-law","grandmother-in-law","daughter-in-law","stepdaughter-in-law",
            "granddaughter-in-law","sister-in-law","half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law","niece-in-law","great-niece-in-law",'mommy','she','her']
            # ตรวจสอบคำในความคิดเห็น
            for keyword in male_keywords:
                if keyword in comment :
                    return "เพศชาย"
            for keyword in female_keywords:
                if keyword in comment:
                    return "เพศหญิง"
            return "ไม่สามารถระบุได้"
        k3=[]
        k4=[]
        for i in data_raddit['comments']:
            k3.append(detect_person_en(str(i)))
            if detect_person_en(str(i)) == "เล่าประสบการณ์ตัวเอง":
                k4.append("ไม่สามารถระบุได้")
            elif detect_person_en(str(i)) == "เล่าประสบการณ์คนอื่น":
                k4.append(detect_gender_other_en(str(i)))
            elif detect_person_en(str(i)) == "ไม่สามารถระบุได้":
                k4.append(detect_gender_other_en(str(i)))
        symptoms_colcan_en = []
        for i in range(len(list_token_red)):
            list_symptoms_en= []
            for j in range(len(list_token_red[i])):
                for k in cancer_symptoms_en:
                    for l in range(len(cancer_symptoms_en[k])):
                        if list_token_red[i][j] == cancer_symptoms_en[k][l]:
                            list_symptoms_en.append(cancer_symptoms_en[k][0])
            unique_list_symptoms_en = list(OrderedDict.fromkeys(list_symptoms_en))
            if len(unique_list_symptoms_en) > 0:
                symptoms_colcan_en.append(unique_list_symptoms_en)
            else :
                symptoms_colcan_en.append(['No symptoms identified'])
        all_reddit_data['defind_Genden_with_nlp'] = new_colgenden_en
        all_reddit_data['defind_Genden_with_python'] = k4
        all_reddit_data['defind_exp_with_python'] = k3
        all_reddit_data['beha']= behaviour_extion
        all_reddit_data['Treatment'] = Treatment_extion
        comment_label = []
        for text in all_reddit_data['comments']:
            predicted_sentiment = predict_sentiment_thai(text,model_thai=model_eng, tokenizer_Autp_T=tokenizer_Autp_E, device=device)
            comment_label.append(predicted_sentiment)
        all_reddit_data['label'] = comment_label
        # all_reddit_data['use_ful'] = 'non'
        all_reddit_data['symptoms_colcan_en'] = symptoms_colcan_en
        # สร้างเงื่อนไขสำหรับการแก้ไขข้อมูล
        condition = all_reddit_data['defind_cancer_with_nlp'].isin(['breast cancer', 'ovarian cancer', 'cervical cancer','uterine cancer','endometrial cancer'])
        # แก้ไขข้อมูลในคอลัมน์ 'defind_Genden_with_python' ตามเงื่อนไข
        all_reddit_data.loc[condition, 'defind_Genden_with_python'] = 'เพศหญิง'
        label_symptoms_en=all_reddit_data['symptoms_colcan_en'].str.join(sep='*').str.get_dummies(sep='*')
        all_reddit_data = all_reddit_data.join(label_symptoms_en)
        all_reddit_data.to_csv('data_pre.csv', index=False, encoding='utf-8-sig')
        data_show = all_reddit_data.iloc[:,:3]
    sorue_sym_4 = pd.read_csv('soure_url.csv')
    sorue_chack = sorue_sym_4['url'][0]
    if sorue_chack == 'www.facebook.com': 
        _ = ['ยอดไลค์','จำนวนการตอบกลับ','จำนวนคำ']
        data = Data_pre_and_clane
        def sort_data(column_name,how_sort):
            if column_name == 'ยอดไลค์':
                data_show.sort_values('ยอดไลค์', inplace=True, ascending=how_sort)
            elif column_name == 'จำนวนการตอบกลับ':
                data_show.sort_values('จำนวนการตอบกลับ', inplace=True, ascending=how_sort)
            elif column_name == 'จำนวนคำ':
                data_show.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
            else:
                pass
            return data_show
        sort_options = ['ยอดไลค์', 'จำนวนการตอบกลับ', 'จำนวนคำ']
        mylist_name_can = data['defind_cancer_with_nlp'].to_list()
        name_can  = list(dict.fromkeys(mylist_name_can))
        sym_list = data.columns
        symptoms_can = sym_list[16:]
        data_name_sym_have = pd.DataFrame()
        data_value_TH = pd.DataFrame(data={'name_cancarTH':name_can})
        data_symptoms_TH = pd.DataFrame(data={'Key_symptoms_TH':symptoms_can})
        list_100 = list(range(0,100))
        data_name_sym_have['index'] = list_100
        data_name_sym_have = data_name_sym_have.merge(data_value_TH.reset_index(), how='outer')
        data_name_sym_have = data_name_sym_have.merge(data_symptoms_TH.reset_index(), how='outer')
        data_name_sym_have.to_csv('data_name_sym_have.csv',encoding='utf-8-sig')
        #------word cloud---------#
        from pythainlp.tokenize import word_tokenize as to_th_k # เป็นตัวตัดคำของภาษาไทย
        from pythainlp.corpus import thai_stopwords # เป็นคลัง Stop Words ของภาษาไทย
        text_th= ''
        for row in data['comments']: # ให้ python อ่านข้อมูลรีวิวจากทุก row ใน columns 'content'
            text_th = text_th + row.lower() + ' ' # เก็บข้อมูลรีวิวของเราทั้งหมดเป็น String ในตัวแปร text

        wt_th = to_th_k(text_th, engine='newmm') # ตัดคำที่ได้จากตัวแปร text

        path_th = 'THSarabunNew-20240628T045147Z-001\THSarabunNew\THSarabunNew.ttf' # ตั้ง path ไปหา font ที่เราต้องการใช้แสดงผล
        wordcloud_th = WordCloud(font_path = path_th, # font ที่เราต้องการใช้ในการแสดงผล เราเลือกใช้ THSarabunNew
                            stopwords = thai_stopwords(), # stop words ที่ใช้ซึ่งจะโดนตัดออกและไม่แสดงบน words cloud
                            relative_scaling = 0.3,
                            min_font_size = 1,
                            background_color = "white",
                            width=620,
                            height=300,
                            max_words = 500, # จำนวนคำที่เราต้องการจะแสดงใน Word Cloud
                            colormap = 'plasma',
                            scale = 3,
                            font_step = 4,
                            collocations = False,
                            regexp = r"[ก-๙a-zA-Z']+", # Regular expression to split the input text into token
                            margin=2).generate(' '.join(wt_th)) # input คำที่เราตัดเข้าไปจากตัวแปร wt ในรูปแบบ string

        wordcloud_th.to_file("wordcloud.png")
        max_v = []
        min_v = []
        avg_v = []
        for i in range(len(data_show.columns)):
            if i >= 2 :
                max_v.append(max(data_show[data_show.columns[i]].tolist()))
                min_v.append(min(data_show[data_show.columns[i]].tolist()))
                avg_v.append(np.mean(data_show[data_show.columns[i]].tolist()))
        descriptive = pd.DataFrame()
        descriptive[' '] = _
        descriptive['max'] = max_v
        descriptive['min'] = min_v
        descriptive['avg'] = avg_v
        descriptive.to_csv('data_desc.csv',encoding='utf-8-sig')
        sorted_data = sort_data('ยอดไลค์',False)
    elif sorue_chack == 'www.reddit.com':
        import pandas as pd
        from nltk.tokenize import word_tokenize as to_en
        from nltk.stem import WordNetLemmatizer
        data = all_reddit_data
        _ = ['จำนวนคำ']
        def sort_data(column_name,how_sort):
            if column_name == 'จำนวนคำ':
                data_show.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
            else:
                pass
            return data_show
        sort_options = ['จำนวนคำ']
        mylist_name_can = data['defind_cancer_with_nlp'].to_list()
        name_can  = list(dict.fromkeys(mylist_name_can))
        symptoms_can = data.columns[16:]
        data_name_sym_have = pd.DataFrame()
        data_value_TH = pd.DataFrame(data={'cancer_names_en':name_can})
        data_symptoms_TH = pd.DataFrame(data={'Key_symptoms_EN':symptoms_can})
        list_100 = list(range(0,100))
        data_name_sym_have['index'] = list_100
        data_name_sym_have = data_name_sym_have.merge(data_value_TH.reset_index(), how='outer')
        data_name_sym_have = data_name_sym_have.merge(data_symptoms_TH.reset_index(), how='outer')
        data_name_sym_have.to_csv('data_name_sym_have.csv',encoding='utf-8-sig')
        import pandas as pd
        #------word cloud---------#
        # Combine all text into a single string
        text = ' '.join(data['comments'].str.lower())
        # Tokenize the text
        tokens = to_en(text)
        # Remove stop words and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        filtered_words = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
        # Create a word cloud
        wordcloud = WordCloud(
            stopwords=stop_words,
            background_color="white",
            width=620,
            height=300,
            max_words=500,
            colormap='plasma',
            scale=3,
            font_step=4,
            collocations=False,
            margin=2
        ).generate(' '.join(filtered_words))
        # Save the word cloud to a file
        wordcloud.to_file("wordcloud.png")
        max_v = []
        min_v = []
        avg_v = []
        for i in range(len(data_show.columns)):
            if i == 2:
                max_v.append(max(data_show[data_show.columns[i]].tolist()))
                min_v.append(min(data_show[data_show.columns[i]].tolist()))
                avg_v.append(np.mean(data_show[data_show.columns[i]].tolist()))
        descriptive = pd.DataFrame()
        descriptive[' '] = _
        descriptive['max'] = max_v
        descriptive['min'] = min_v
        descriptive['avg'] = avg_v
        descriptive.to_csv('data_desc.csv',encoding='utf-8-sig')
        sorted_data = sort_data('ความยาวของความคิดเห็น',False)
    descriptive =pd.read_csv('data_desc.csv')
    descriptive = descriptive.iloc[:, 1:]
    tables_d = descriptive.to_html(classes='table table-striped', index=False)
    # เรียงลำดับข้อมูลตามค่าเริ่มต้น (like)
    sorted_data.to_csv('sorted_data.csv',encoding='utf-8-sig')
    tables = sorted_data.to_html(classes='table table-striped', index=False)
    x06 = pd.read_csv('ref_pas.csv')
    x06['ref'][0] = 10
    x16 = x06['ref'][0]
    x30 = x06['refy'][0]
    x06.to_csv('ref_pas.csv',index=False)
    return  render_template('output2.html',tables=[tables],sort_options=sort_options,skills=name_can,symptoms=symptoms_can
                           ,tables_descript=[tables_d] ,count_user = count_user,count_comment=count_comment,number_of_rows=number_of_rows
                           ,number_of_columns=number_of_columns,x=x16,y=x30)     
    



@server.route('/sort', methods=['POST','GET'])
def sort():
    soure_b = pd.read_csv('soure_url.csv')
    soure = soure_b['url'][0]
    data = pd.read_csv("sorted_data.csv", encoding='utf-8-sig')
    data = data.iloc[:,1:]
    number_of_rows = len(data)
    number_of_columns = len(data.columns)
    count_user = len(set(data.iloc[:,0]))
    count_comment = len(set(data.iloc[:,1]))
    if soure == 'www.facebook.com':
        def sort_data(column_name,how_sort):
            if column_name == 'ยอดไลค์':
                data.sort_values('ยอดไลค์', inplace=True, ascending=how_sort)
            elif column_name == 'จำนวนการตอบกลับ':
                data.sort_values('จำนวนการตอบกลับ', inplace=True, ascending=how_sort)
            elif column_name == 'จำนวนคำ':
                data.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
            else:
                pass
            return data
        sort_options = ['ยอดไลค์','จำนวนการตอบกลับ','จำนวนคำ']
        data_cancer= pd.read_csv('data_name_sym_have.csv')
        name_can = data_cancer['name_cancarTH'].dropna().to_list()
        symptoms_can = data_cancer['Key_symptoms_TH'].dropna().to_list() 
    elif soure == 'www.reddit.com':
        def sort_data(column_name,how_sort):
            if column_name == 'จำนวนคำ':
                data.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
            else:
                pass
            return data
        sort_options = ['จำนวนคำ']
        data_cancer= pd.read_csv('data_name_sym_have.csv')
        name_can = data_cancer['cancer_names_en'].dropna().to_list()
        symptoms_can = data_cancer['Key_symptoms_EN'].dropna().to_list()
    descriptive=pd.read_csv('data_desc.csv',encoding='utf-8-sig')
    descriptive = descriptive.iloc[:, 1:]
    tables_d = descriptive.to_html(classes='table table-striped', index=False)
    # รับค่าคอลัมน์ที่เลือกจาก dropdownlist
    column_name = request.form['sort_column']
    how_sort_clk = request.form['HOW']
    if how_sort_clk == 'มากไปน้อย' :
        how_sort = False
    elif how_sort_clk == 'น้อยไปมาก':
        how_sort = True
    # เรียงลำดับข้อมูลตามคอลัมน์ที่เลือก
    sorted_data = sort_data(column_name,how_sort)
    tables = sorted_data.to_html(classes='table table-striped', index=False)
    # ส่งข้อมูลไปยังเทมเพลต HTML
    x07 = pd.read_csv('ref_pas.csv')
    x17 = x07['ref'][0]
    x30 = x07['refy'][0]
    return render_template('output2.html', tables=[tables], sort_options=sort_options,skills=name_can,symptoms=symptoms_can
                           ,tables_descript=[tables_d],number_of_rows=number_of_rows,number_of_columns=number_of_columns,count_user = count_user,count_comment=count_comment,
                           x=x17,y=x30)

@server.route('/wordcloud.png')
def wordcloud():
    return send_from_directory('.', 'wordcloud.png')
#number_of_rows=number_of_rows, number_of_columns=number_of_columns,

@server.route("/ajax_add",methods=["POST","GET"])
def ajax_add():
    #========================================== ส่วนตัวรับข้อมูลเพิ่ม/ลด โรคเเละอาการ ======================================
    if request.method == 'POST':
        # ข้อมูลที่ได้รับจากการกรอง
        import pandas as pd
        skill = request.form['skill']
        symptom  = request.form['symptom']
        sum_ch = request.form['sum_ch']
        try:
            skill = skill.replace(',  ', ', ')
            name_cancer=skill.split(', ')
        except:
            name_cancer=skill.split(', ')
        try:
            symptom = symptom.replace(',  ', ', ')
            name_symptom = symptom.split(', ')
        except:
            name_symptom = symptom.split(', ')
        # print(name_symptom[-1])
        # print(type(name_symptom[-1]))
        # name_symptom = name_symptom[:-1]
        soure_b = pd.read_csv('soure_url.csv')
        soure = soure_b['url'][0]
        if name_cancer ==[''] or name_cancer == [] or name_cancer is None :
            if soure == 'www.facebook.com':
                data_sy_na = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
                data_defind = data_sy_na[['name_cancarTH']]
                name_cancer = data_defind['name_cancarTH'].to_list()
            elif soure == 'www.reddit.com':
                data_sy_na = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
                data_defind = data_sy_na[['cancer_names_en']]
                cancer_names_en = data_defind['cancer_names_en'].to_list()
        if name_symptom ==[''] or name_symptom == [] or name_symptom is None :
            if soure == 'www.facebook.com':
                data_sy_na = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
                data_defind = data_sy_na[['Key_symptoms_TH']]
                name_symptom = data_defind['Key_symptoms_TH'].to_list()
            elif soure == 'www.reddit.com':
                data_sy_na = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
                data_defind = data_sy_na[['Key_symptoms_EN']]
                data_k_list = data_defind['Key_symptoms_EN'].to_list()
        # ข้อมูลอาการเพิ่มเติม
        # c_submit= request.form['custId']
        if soure == 'www.facebook.com':
            data_defu = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')
            # cancer_names_from_csv = [item for item in data_defu['name_cancarTH'].tolist() if not pd.isna(item)]
            # symptom_names_from_csv = [item for item in data_defu['Key_symptoms_TH'].tolist() if not pd.isna(item)]

            data_value_th = pd.DataFrame({
                'name_cancarTH_se': [item.strip() for item in name_cancer ] #+ cancer_names_from_csv
            })

            data_symptoms_th = pd.DataFrame({
                'Key_symptoms_TH':[item.strip() for item in name_symptom ] #+ symptom_names_from_csv
            })

            data_symptoms_with_values = data_symptoms_th.merge(data_defu[['Key_symptoms_TH', 'Values_symptoms_TH']].dropna(),
                                                            how='left', on='Key_symptoms_TH')

            symptoms_with_missing_values = data_symptoms_with_values[
                data_symptoms_with_values['Values_symptoms_TH'].isna()]['Key_symptoms_TH'].tolist()

            data_symptoms_with_values['Values_symptoms_TH'] = data_symptoms_with_values['Values_symptoms_TH'].fillna(
                data_symptoms_with_values['Key_symptoms_TH'].apply(lambda x: f"['{x}']"))

            all_data = pd.DataFrame({'index': range(100)})
            all_data = all_data.merge(data_value_th.reset_index(), how='outer')
            all_data = all_data.merge(data_symptoms_with_values.reset_index(), how='outer')
            all_data = all_data.dropna(how='all')
        elif soure == 'www.reddit.com':
            data_defu = pd.read_csv('name_cancer_and_symptoms_lowercase.csv')

            # # ลบค่า NaN ออกจากลิสต์ก่อนสร้าง DataFrame
            # cancer_names_from_csv = [item for item in data_defu['cancer_names_en'].tolist() if not pd.isna(item)]
            # symptom_names_from_csv = [item for item in data_defu['Key_symptoms_EN'].tolist() if not pd.isna(item)]

            data_value_th = pd.DataFrame({
                'cancer_names_en_se': [s.lower() for s in name_cancer]+['Unable to identify / not sure if it is']#+ cancer_names_from_csv
            })

            data_symptoms_th = pd.DataFrame({
                'Key_symptoms_EN': [s.lower() for s in name_symptom]+['No symptoms identified']#+ symptom_names_from_csv
            })

            data_symptoms_with_values = data_symptoms_th.merge(data_defu[['Key_symptoms_EN', 'Valuessymptoms_EN']].dropna(),
                                                            how='left', on='Key_symptoms_EN')

            symptoms_with_missing_values = data_symptoms_with_values[
                data_symptoms_with_values['Valuessymptoms_EN'].isna()]['Key_symptoms_EN'].tolist()

            data_symptoms_with_values['Valuessymptoms_EN'] = data_symptoms_with_values['Valuessymptoms_EN'].fillna(
                data_symptoms_with_values['Key_symptoms_EN'].apply(lambda x: f"['{x}']"))

            all_data = pd.DataFrame({'index': range(100)})
            all_data = all_data.merge(data_value_th.reset_index(), how='outer')
            all_data = all_data.merge(data_symptoms_with_values.reset_index(), how='outer')

            # ลบแถวที่มีค่า NaN ทั้งหมด
            all_data = all_data.dropna(how='all')
        all_data.to_csv('all_data_nameandsym.csv', index=False, encoding='utf-8-sig')
        sorue = soure
        #===================================================================================================
        # update ตาราง 
        if sorue == 'www.facebook.com':
            # ระบุโรค
            name_cancar_and_symptoms_pd = pd.read_csv('all_data_nameandsym.csv')
            name_cancar_list  = name_cancar_and_symptoms_pd['name_cancarTH_se'].tolist()
            name_cancar = [item for item in name_cancar_list if not(pd.isnull(item)) == True]
            # ระบุอาการ
            symptoms_pd = name_cancar_and_symptoms_pd[['Key_symptoms_TH','Values_symptoms_TH']].dropna()
            data_k_list = symptoms_pd['Key_symptoms_TH'].tolist()
            data_V_list = symptoms_pd['Values_symptoms_TH'].tolist()
            symptoms = {}
            for list_item in range(len(data_V_list)):
                split_t = ast.literal_eval(data_V_list[list_item])
                split_t = [n.strip() for n in split_t]
                symptoms[data_k_list[list_item]]=split_t
        #====================================================================================================
            #เเปลงหัวตาราง
            data = pd.read_csv('data_tokenizer.csv', encoding='utf-8-sig')
            comment=data.groupby('name').sum().reset_index()
            comment = comment.drop(['Unnamed: 0'], axis=1)
            # Apply ast.literal_eval to each element in the 'token' column
            comment['token'] = comment['token'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            comment['token_2g'] = comment['token_2g'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            comment['token_3g'] = comment['token_3g'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            # Access the list of tokens
            list_token = comment['token'].tolist()

            def analyze_comments_multi_level(comment, name_cancar):
                """วิเคราะห์ความคิดเห็นเพื่อหาโรคโดยใช้ list_token, token_2gam, และ token_3gam
                โดยตรวจสอบว่า token_pair หรือ token_triple ตรงกับ name_cancar โดยตรง
                """
                list_token_1g = comment['token']
                list_token_2gam = comment['token_2g']
                list_token_3gam = comment['token_3g']
                list_cancer = []
                # วิเคราะห์ list_token
                for token in list_token_1g:
                    if token in ("มะเร็ง", "โรคมะเร็ง", "โรค", "หัวใจ"):
                        list_cancer.append(token)
                for token in list_token_1g:
                    if token in ("เบาหวาน", "อ้วน", "ตับแข็ง"):
                        list_cancer.append("โรค" + token)
                for token in list_token_1g:
                    if token == "ความดัน":
                        list_cancer.append("โรคความดันโลหิต")
                for token in list_token_1g:
                    if token in name_cancar:
                        list_cancer.append(token)
                # วิเคราะห์ token_2gam
                for token_pair in list_token_2gam:
                    if token_pair in name_cancar:
                        list_cancer.append(token_pair)
                # วิเคราะห์ token_3gam
                for token_triple in list_token_3gam:
                    if token_triple in name_cancar:
                        list_cancer.append(token_triple)
                unique_list = list(OrderedDict.fromkeys(list_cancer))
                list_define_cancer = [disease for disease in unique_list if disease in name_cancar]
                if len(list_define_cancer) > 1:
                    return 'เล่ามากกว่า 2 โรค'
                elif len(list_define_cancer) == 1:
                    return list_define_cancer[0]
                else:
                    return 'ไม่สามารถระบุได้/ไม่มั่นใจว่าเป็น'
            #----------------------------------------------------------------------
            # สร้าง list เพศ 
            Genden = {'ชาย':['พ่อ','บิดา','พี่ชาย','น้องชาย','ลูกชาย','สามี','พัว','ผัว','ปู่','ตา','คุณปู่','คุณตา','คุณพ่อ',
                            'ปู่ทวด','ตาทวด','ลุง','อาหนู','คุณอา','คุณลุง','หลายชาย','ลูกเขย','เขย','พี่เขย','น้องเขย',
                            'พ่อตา','พ่อผม','พ่อหนู','พ่อพม','ชาย','หนุ่ม','ลช.','ผ่อ','ชย.','น้าชาย','ผ่อตา','หน.']
                    ,'หญิง':['แม่','เเม่','คุณแม่','มารดา','พี่สาว','น้องสาว','ลูกสาว','ภรรยา','เมีย','ย่า','ยาย','คุณย่า',
                            'คุณยาย','คุณเเม่','ย่าทวด','ยายทวด','ป้า','น้า','คุณป้า','คุณน้า','หลายสาว','ลูกสะใถ้',
                            'ลูกสะใภ้','สะใภ้','พี่สะใภ้','น้องสะใภ้','เเม่ผม','เเม่หนู','เเม่พม','แม่ผม','แม่หนู','แม่พม','สาว','หญิง','ก้อน','คลำ']}
            # หาเพศ โดยใช้nlp
            new_colgenden=[]
            list_genden=[]
            for i in range(len(list_token)):
                for j in range(len(Genden['หญิง'])):
                    for k in range(len(list_token[i])):
                        if (list_token[i][k] == Genden['ชาย'][j]):
                            list_genden.append('เพศชาย')
                        elif(list_token[i][k] == Genden['หญิง'][j]):
                            list_genden.append('เพศหญิง')
                genden_list =[]
                genden_list = list(OrderedDict.fromkeys(list_genden)) # ลบคำซ้ำ
                #-------------------------------------------------------------------
                list_define_genden = []
                if len(genden_list) > 0 :
                    if len(genden_list) == 2:
                        list_define_genden.append('เล่าทั้งสองเพศ')
                    elif len(genden_list)==1:
                        list_define_genden.append(genden_list[0])
                elif len(genden_list)==0:
                    list_define_genden.append('ไม่สามารถระบุได้/ไม่มั่นใจว่าเป็น')
                genden_list_de =[]
                genden_list_de = list(OrderedDict.fromkeys(list_define_genden))
                new_colgenden.append(genden_list_de[0])

            # หาอาการโดย nlp
            # symptoms_colcan = []
            # for i in range(len(list_token)):
            #     list_symptoms= []
            #     for j in range(len(list_token[i])):
            #         for k in symptoms:
            #             for l in range(len(symptoms[k])):
            #                 if list_token[i][j] == symptoms[k][l]:
            #                     list_symptoms.append(symptoms[k][0])
            #     unique_list_symptoms = list(OrderedDict.fromkeys(list_symptoms))
            #     if len(unique_list_symptoms) > 0:
            #         symptoms_colcan.append(unique_list_symptoms)
            #     else :
            #         symptoms_colcan.append(['ไม่มีการระบุอาการ'])
            def analyze_symptoms_multi_level(comment, ngram_cancer_symptoms):
                """วิเคราะห์ข้อมูล comment เพื่อหาอาการโดยใช้ list_token_1g, list_token_2gam, และ list_token_3gam"""
                list_token_1g = comment['token']
                list_token_2gam = comment['token_2g']
                list_token_3gam = comment['token_3g']
                list_symptoms_en = []

                # วิเคราะห์ list_token_1g
                for token in list_token_1g:
                    for symptom, tokens in ngram_cancer_symptoms.items():
                        if token in tokens:
                            list_symptoms_en.append(symptom)

                # วิเคราะห์ list_token_2gam
                for token_pair in list_token_2gam:
                    for symptom, tokens in ngram_cancer_symptoms.items():
                        if token_pair in tokens:
                            list_symptoms_en.append(symptom)

                # วิเคราะห์ list_token_3gam
                for token_triple in list_token_3gam:
                    for symptom, tokens in ngram_cancer_symptoms.items():
                        if token_triple in tokens:
                            list_symptoms_en.append(symptom)

                unique_list_symptoms_en = list(OrderedDict.fromkeys(list_symptoms_en))
                print(unique_list_symptoms_en)

                if len(unique_list_symptoms_en) > 0:
                    return unique_list_symptoms_en
                else:
                    return ['ไม่มีการระบุอาการ']

            #สร้างฟังชั่นหาเพศกับใครเป็นคนพูด
            def detect_person(comment):
            # คำที่ใช้ตรวจสอบว่ามีใครเป็นคนอยู่ในความคิดเห็น
                other = ['พ่อ','บิดา','พี่ชาย','น้องชาย','ลูกชาย','สามี','พัว','ผัว','ปู่','ตา','คุณปู่','คุณตา','คุณพ่อ',
                        'ปู่ทวด','ตาทวด','ลุง','อาหนู','คุณอา','คุณลุง','หลายชาย','ลูกเขย','เขย','พี่เขย','น้องเขย',
                        'พ่อตา','พ่อผม','พ่อหนู','พ่อพม','ชาย','หนุ่ม','แม่','เเม่','คุณแม่','มารดา','พี่สาว','น้องสาว',
                        'ลูกสาว','ภรรยา','เมีย','ย่า','ยาย','คุณย่า','คุณยาย','คุณเเม่','ย่าทวด','ยายทวด','ป้า',
                        'น้า','คุณป้า','คุณน้า','หลายสาว','ลูกสะใถ้','ลูกสะใภ้','สะใภ้','พี่สะใภ้','น้องสะใภ้','เเม่ผม','เเม่หนู','เเม่พม','แม่ผม','แม่หนู','แม่พม','สาว','หญิง']
                myself = ['ผมเป็น','ผมเอง','กระผม','พมเป็น','พมเอง','กระพม','หนูเอง','หนู','ดิฉัน','ตัวเอง','ก้อน','คลำ']
                # ตรวจสอบคำในความคิดเห็น
                for keyword in  other:
                    if keyword in comment and "อาการ" not in comment :
                        return 'เล่าประสบการณ์คนอื่น'
                for keyword in  myself:
                    if keyword in comment:
                        return 'เล่าประสบการณ์ตัวเอง'
                # หากไม่พบคำที่บ่งบอกถึงคน
                return 'ไม่สามารถระบุได้'
            def detect_gender_other(comment):
                # คำที่บ่งบอกถึงเพศชาย
                male_keywords = ['พ่อ','บิดา','พี่ชาย','น้องชาย','ลูกชาย','สามี','พัว','ผัว','ปู่','ตา','คุณปู่','คุณตา','คุณพ่อ',
                                'ปู่ทวด','ตาทวด','ลุง','อาหนู','คุณอา','คุณลุง','หลายชาย','ลูกเขย','เขย','พี่เขย','น้องเขย',
                                'พ่อตา','พ่อผม','พ่อหนู','พ่อพม','ชาย','หนุ่ม']
                # คำที่บ่งบอกถึงเพศหญิง
                female_keywords = ['แม่','เเม่','คุณแม่','มารดา','พี่สาว','น้องสาว','ลูกสาว','ภรรยา','เมีย','ย่า','ยาย','คุณย่า','คุณยาย','คุณเเม่','ย่าทวด','ยายทวด','ป้า',
                                'น้า','คุณป้า','คุณน้า','หลายสาว','ลูกสะใถ้','ลูกสะใภ้','สะใภ้','พี่สะใภ้','น้องสะใภ้','เเม่ผม','เเม่หนู','เเม่พม','แม่ผม','แม่หนู','แม่พม','สาว','หญิง']
                # ตรวจสอบคำในความคิดเห็น
                for keyword in male_keywords:
                    if keyword in comment and "อาการ" not in comment :
                        return "เพศชาย"
                for keyword in female_keywords:
                    if keyword in comment:
                        return "เพศหญิง"
                # หากไม่พบคำที่บ่งบอกถึงเพศ
                return "ไม่สามารถระบุได้"
            def detect_gender_self(comment):
                # คำที่บ่งบอกถึงเพศชาย
                male_keywords = ['ผมเป็น','ครับ','ผมเอง','คับ','กระผม','พมเป็น','พมเอง','กระพม']
                # คำที่บ่งบอกถึงเพศหญิง
                female_keywords = ['ค่ะ','คะ','หนูเอง','หนูเป็น','ดิฉัน','ก้อน','คลำ']
                # ตรวจสอบคำในความคิดเห็น
                for keyword in male_keywords:
                    if keyword in comment:
                        return "เพศชาย"
                for keyword in female_keywords:
                    if keyword in comment:
                        return "เพศหญิง"
                # หากไม่พบคำที่บ่งบอกถึงเพศ
                return "ไม่สามารถระบุได้"
            #เเบ่งเพศเเละใครเล่าโดยใช้ python
            k1=[]
            k2=[]
            for i in comment['comments']:
                k1.append(detect_person(str(i)))
                if detect_person(str(i)) == 'เล่าประสบการณ์คนอื่น':
                    k2.append(detect_gender_other(str(i)))
                elif detect_person(str(i)) == 'เล่าประสบการณ์ตัวเอง':
                    k2.append(detect_gender_self(str(i)))
                elif detect_person(str(i)) == 'ไม่สามารถระบุได้':
                    k2.append(detect_gender_self(str(i)))
            # #หาพฤติกรรม
            behaviour_value ={'นอนไม่เป็นเวลา':['นอนดึก','นอนไม่เป็นเวลา',"อดนอน"],
                              'กินอาหารไม่ถูกสุขภาพ':['กินของทอด','กินของมัน','ไม่กินผัก','เค็มจัด','กินเค็ม','หวานจัด','กินหวาน','ไม่คุมอาหาร','อาหารทอด','ทอดอาหาร'
                                ,'กินของหมักดอง','กินของหมักดอง','กินของปิ้ง','กินของย่าง','ปิ้งย่าง'],
                                'ดื่มน้ำหวาน':['กินน้ำอัดลม', 'น้ำหวาน'],
                                'ไม่ออกกำลังกาย':['ไม่ออกกำลังกาย','ไม่เล่นกีฬา','ไม่ชอบออกกำลังกาย','กำลังกายไม่ค่อย','กำลังกายไม่ออก','ว่ายน้ำ', 'วิ่ง', 'ปั่นจักรยาน', 'กระโดดเชือก'],
                                'นอนตรงเวลา':['ตอนแต่วัน','นอนตรงเวลา'],
                                'กินอาหารสุขภาพ':['ไม่กินของ','กินผัก','ไม่กินเค็ม','ไม่กินหวาน','คุมอาหาร','ดื่มน้ำ','ลดหวาน','ลดเค็ม','คุมการกิน'],
                                'ออกกำลังกาย':['ออกกำลังกาย','เล่นกีฬา','ชอบออกกำลังกาย'],
                                'ไม่สูบบุหรี่':['ไม่สูบ','ไม่สูบบุหรี่','ไม่บุหรี่','เลิกสูบ','งดบุหรี่'],
                                'สูบบุหรี่':['ชอบสูบ','สูบบุหรี่','บุหรี่'],
                                'ไม่ดื่มเหล้า':['ไม่ดื่ม','ไม่ดื่มเหล้า','ไม่เหล้า','งดเหล้า'],
                                'ดื่มเหล้า':['ดื่มเหล้า','กินเหล้า','ดื่มแอลกอฮอล์'],
                                'อยู่ใกล้บุหรี่':['ใกล้คนสูบ','ครอบครัวสูบ','บุหรี่มือสอง'],
                                'อยู่ใกล้แหล่งมลผิษ':['ควัน','มลผิษ','ฝุ่นพง','ฝุ่น','ก่อสร้าง'],
                                "เครียด":["เครียด","อาการเครียด"],
                                "ทำงานหนัก":["ทำงานหนัก"]}
            behaviour_key = ['นอนไม่เป็นเวลา','กินอาหารไม่ถูกสุขภาพ','ดื่มน้ำหวาน','ไม่ออกกำลังกาย','นอนตรงเวลา','กินอาหารสุขภาพ','ออกกำลังกาย','ไม่สูบบุหรี่','สูบบุหรี่','ไม่ดื่มเหล้า','ดื่มเหล้า'
            ,'อยู่ใกล้บุหรี่','อยู่ใกล้แหล่งมลผิษ',"เครียด"]
            behaviour_extion = []
            for i in range(len(comment)):
                behaviour_list=[]
                for j in behaviour_key:
                    for lol in behaviour_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in comment['token'].iloc[i]:
                            behaviour_list.append(j)
                        else:
                            behaviour_list.append('ไม่สามารถระบุได้')
                for j in behaviour_key:
                    for lol in behaviour_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in comment['token_2g'].iloc[i]:
                            behaviour_list.append(j)
                        else:
                            behaviour_list.append('ไม่สามารถระบุได้')
                for j in behaviour_key:
                    for lol in behaviour_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in comment['token_3g'].iloc[i]:
                            behaviour_list.append(j)
                        else:
                            behaviour_list.append('ไม่สามารถระบุได้')
                behaviour_list =list(set(behaviour_list))
                remove_negations(behaviour_list)
                if len(behaviour_list) >= 2:
                    behaviour_list.remove('ไม่สามารถระบุได้')
                behaviour_extion.append(behaviour_list)
            # วิธีการรักษา
            Treatment_value ={'Surgery':['ผ่าตัด','Surgery','ผ่า','operation', 'incision', 'surgery','บายพาส','สวนหัวใจ','ลดขนาดกระเพาะอาหาร','ตัดกระเพาะอาหาร'],
                        'Chemotherapy':['chemotherapy','chemo','คีโม','เคมี','herceptin','บำบัดรักษา'],
                        'Radiation Therapy':['radiation','ฉายแสง','เเร่','ฉายเเสง'],
                        'Targeted Therapy':['targeted','เป้า'],
                        "Diet control": ["ลดแป้ง","ลดน้ำตาล","ลดไขมัน", "กินผัก" ,"กินผลไม้" ,"กินธัญพืช", "เพิ่มใยอาหาร","ลดคอเลสเตอรอล","ลดปริมาณอาหาร","ลดปริมาณ"
                                    ,"ปรับเปลี่ยนพฤติกรรมการกิน", "เปลี่ยนการกิน","ลดโซเดียม", "ลดโพแทสเซียม",
                                    "reducestarch", "reducesugar", "reducefat", "eatvegetables", "eatfruits", "eatgrains", "increasedietary", "reducecholesterol", "Reducefood", "reduceintake"
                                    , "changeeat",  "reducesodium", "reducepotassium"],
                        "Drug use": ["อินซูลิน","ยาลดน้ำตาล","ยาลด","ยาลดความดัน", "ยาลดไขมัน","ยาลดความอยากอาหาร","กินยา",
                                    "insulin", "drugs", "diet drugs", "lowering", "suppressants", "medicine"],
                        "Self-care": ["ตรวจระดับน้ำตาล", "ดูแลเท้า","ลดเครียด","ลดความเครียด","พักผ่อนให้เพียงพอ","พักผ่อน",
                                    "bloodsugar", "care", "reducestress", "rest", "relax"],
                        }
            Treatment_key = ['Surgery','Chemotherapy','Radiation Therapy','Targeted Therapy','Diet control',"Drug use","Self-care"]
            Treatment_extion = []
            for i in range(len(comment)):
                Treatment_list=[]
                for j in Treatment_key:
                    for lol in Treatment_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in comment['token'].iloc[i]:
                            Treatment_list.append(j)
                        elif lol in comment['token_2g'].iloc[i]:
                            Treatment_list.append(j)
                        elif lol in comment['token_3g'].iloc[i]:
                            Treatment_list.append(j)
                        else:
                            Treatment_list.append('ไม่สามารถระบุได้')
                Treatment_list =list(set(Treatment_list))
                if len(Treatment_list) >= 2:
                    Treatment_list.remove('ไม่สามารถระบุได้')
                Treatment_extion.append(Treatment_list)
            # แยกความคิดเห็น
            # text_use = comm็๋ent['comments'].tolist()
            # labels_use = []
            # for i in text_use:
            #     text_to_predict = i # Replace with your input
            #     predicted_sentiment = predict_sentiment(text_to_predict, model, tokenizer_Autp, device)
            #     labels_use.append(predicted_sentiment)
            Data_pre_and_clane = comment
            Data_pre_and_clane['defind_cancer_with_nlp'] = Data_pre_and_clane.apply(lambda row_1: analyze_comments_multi_level(row_1, name_cancar), axis=1)
            Data_pre_and_clane['use_ful'] = 'non'
            Data_pre_and_clane['defind_exp_with_python'] = k1
            Data_pre_and_clane['defind_Genden_with_nlp'] = new_colgenden
            Data_pre_and_clane['defind_Genden_with_python'] = k2
            Data_pre_and_clane['beha'] = behaviour_extion
            Data_pre_and_clane['Treatment'] = Treatment_extion
            comment_label=[]
            for text in Data_pre_and_clane['comments']:
                predicted_sentiment = predict_sentiment_thai(text,model_thai=model_thai, tokenizer_Autp_T=tokenizer_Autp_T, device=device)
                comment_label.append(predicted_sentiment)
            Data_pre_and_clane['label'] = comment_label
            # Data_pre_and_clane['label'] = 'non'
            Data_pre_and_clane['symptoms_colcan_en']=Data_pre_and_clane.apply(lambda row_1: analyze_symptoms_multi_level(row_1, symptoms), axis=1)
            label_symptoms=Data_pre_and_clane['symptoms_colcan_en'].str.join(sep='*').str.get_dummies(sep='*')
            Data_pre_and_clane=Data_pre_and_clane.join(label_symptoms)
            Data_pre_and_clane.to_csv('data_pre.csv', index=False, encoding='utf-8-sig')
        elif sorue == 'www.reddit.com':
            import pandas as pd
            data = pd.read_csv('data_tokenizer.csv', encoding='utf-8-sig')
            count_user = len(set(data['name']))
            count_comment = len(set(data['comments']))
            # ระบุโรค
            name_cancar_and_symptoms_pd = pd.read_csv('all_data_nameandsym.csv')
            name_cancar_list  = name_cancar_and_symptoms_pd['cancer_names_en_se'].tolist()
            cancer_names_en = [item for item in name_cancar_list if not(pd.isnull(item)) == True]
            # ระบุอาการ
            symptoms_pd = name_cancar_and_symptoms_pd[['Key_symptoms_EN','Valuessymptoms_EN']].dropna()
            data_k_list = symptoms_pd['Key_symptoms_EN'].tolist()
            data_V_list = symptoms_pd['Valuessymptoms_EN'].tolist()
            cancer_symptoms_en = {}
            for list_item in range(len(data_V_list)):
                split_t = ast.literal_eval(data_V_list[list_item])
                split_t = [n.strip() for n in split_t]
                cancer_symptoms_en[data_k_list[list_item]]=split_t
            data_raddit=data.groupby('name').sum().reset_index()
            data_raddit = data_raddit.drop(['Unnamed: 0'], axis=1)
            data_raddit['token'] = data_raddit['token'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            data_raddit['token_2g'] = data_raddit['token_2g'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            data_raddit['token_3g'] = data_raddit['token_3g'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            list_token_red = data_raddit['token'].tolist()
        #------------------------------------------------------------------------------
            def analyze_reddit_data_multi_level(data_raddit, cancer_names_en):
                """วิเคราะห์ข้อมูล Reddit เพื่อหาโรคโดยใช้ list_token, token_2gam, และ token_3gam"""
                list_token_red = data_raddit['token']
                list_token_2gam_red = data_raddit['token_2g']
                list_token_3gam_red = data_raddit['token_3g']
                list_cancer_en = []
                # วิเคราะห์ list_token
                for token in list_token_red:
                    if token in ('cancer', 'disease') and i > 0:
                        list_cancer_en.append(tokens[i - 1] + ' ' + token)

                for token in list_token_red:
                    if token in cancer_names_en:
                        list_cancer_en.append(token)
                # วิเคราะห์ token_2gam
                for token_pair in list_token_2gam_red:
                    if token_pair in cancer_names_en:
                        list_cancer_en.append(token_pair)
                # วิเคราะห์ token_3gam
                for token_triple in list_token_3gam_red:
                    if token_triple in cancer_names_en:
                        list_cancer_en.append(token_triple)

                unique_list_en = list(OrderedDict.fromkeys(list_cancer_en))
                list_define_cancer_en = [disease for disease in unique_list_en if disease in cancer_names_en]

                if len(list_define_cancer_en) > 1:
                    return 'Tell more than 2 diseases.'
                elif len(list_define_cancer_en) == 1:
                    return list_define_cancer_en[0]
                else:
                    return 'Unable to identify / not sure if it is'
    #-----------------------------------
            all_reddit_data=data_raddit
            all_reddit_data['defind_cancer_with_nlp'] = all_reddit_data.apply(lambda row_2: analyze_reddit_data_multi_level(row_2, cancer_names_en), axis=1)
        #-----------------------------------
            data_value_red =[]
            for i in range(len(list_token_red)):
                if len(list_token_red[i]) >= 50 :
                    data_value_red.append('maybe_useful')
                else:
                    data_value_red.append('Not useful or not giving too much information')
            new_colgenden_en=[]
            list_genden_en=[]
            #เเบ่งเพศ
            Genden_en = {'Male':['dad', "father","stepfather","grandfather","great-grandfather",
            "husband","boyfriend","fiancé","son","stepson","grandson","great-grandson","brother",
            "half-brother","stepbrother","uncle","great-uncle","nephew","great-nephew",
            "father-in-law","stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law",
            "grandson-in-law","brother-in-law","half-brother-in-law","stepbrother-in-law","uncle-in-law",
            "great-uncle-in-law","nephew-in-law","great-nephew-in-law","godfather","stepfather",
            "godson","foster father","foster son","stepson","stepbrother","half-brother","great-uncle",
            "great-nephew","first cousin","second cousin","third cousin","father-in-law",
            "stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law","grandson-in-law",
            "brother-in-law","half-brother-in-law","stepbrother-in-law","uncle-in-law","great-uncle-in-law",
            "nephew-in-law","great-nephew-in-law"'he','him','his','boy'],
            'Female':["mother",'mom',"stepmother","grandmother","great-grandmother","wife",
            "girlfriend","fiancée","daughter","stepdaughter","granddaughter","great-granddaughter",
            "sister","half-sister","stepsister","aunt","great-aunt","niece","great-niece",
            "mother-in-law","stepmother-in-law","grandmother-in-law","daughter-in-law",
            "stepdaughter-in-law","granddaughter-in-law","sister-in-law",
            "half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law",
            "niece-in-law","great-niece-in-law","cousin-in-law","godmother","stepmother",
            "goddaughter","foster mother","foster daughter","stepdaughter","stepsister",
            "half-sister","great-aunt","great-niece","mother-in-law","stepmother-in-law","grandmother-in-law",
            "daughter-in-law","stepdaughter-in-law","granddaughter-in-law","sister-in-law",
            "half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law",
            "niece-in-law","great-niece-in-law",'she','her','hers','mommy','girl']}
            for i in range(len(list_token_red)):
                for j in range(len(Genden_en['Female'])):
                    for k in range(len(list_token_red[i])):
                        if (list_token_red[i][k] == Genden_en['Male'][j]):
                            list_genden_en.append('เพศชาย')
                        elif(list_token_red[i][k] == Genden_en['Female'][j]):
                            list_genden_en.append('เพศหญิง')
                genden_list_en =[]
                genden_list_en = list(OrderedDict.fromkeys(list_genden_en)) # ลบคำซ้ำ
                #-------------------------------------------------------------------
                list_define_genden_en = []
                if len(genden_list_en) > 0 :
                    if len(genden_list_en) == 2:
                        list_define_genden_en.append('เล่าทั้งสองเพศ')
                    elif len(genden_list_en)==1:
                        list_define_genden_en.append(genden_list_en[0])
                elif len(genden_list_en)==0:
                    list_define_genden_en.append('ไม่สามารถระบุได้')
                genden_list_de_en =[]
                genden_list_de_en = list(OrderedDict.fromkeys(list_define_genden_en))
                new_colgenden_en.append(genden_list_de_en[0])
            def detect_person_en(comment):
                # คำที่ใช้ตรวจสอบว่ามีใครเป็นคนอยู่ในความคิดเห็น
                other = ['you','your','we','they','he','she','him','her','it']
                myself = ['my','myself']
                for keyword in  myself:
                    if keyword in comment:
                        return "เล่าประสบการณ์ตัวเอง"
                for keyword in  other:
                    if keyword in comment:
                        return "เล่าประสบการณ์คนอื่น"
                # หากไม่พบคำที่บ่งบอกถึงคน
                return "ไม่สามารถระบุได้"
            # #หาพฤติกรรม
            behaviour_value = {
                    'Irregular Sleep': ['staying up late', 'inconsistent sleep schedule', 'not getting enough sleep','no sleep','not sleep'],
                    'Unhealthy Diet': ['beer','coke','salty food','eating a lot of fried food', 'eating a lot of fatty food', 'not eating vegetables', 'eating very salty food', 'eating very sweet food', 'not watching what I eat', 'fried foods', 'deep-fried foods', 'pickled foods', 'fermented foods', 'grilled foods', 'roasted foods', 'grilling', 'barbecuing'],
                    'Sugary Drinks': ['drinking soda', 'Sweet drinks'],
                    'Lack of Exercise': ['not exercising', 'Not playing sports', 'Not liking exercise', 'Not exercising much', 'Having trouble exercising', 'Swimming', 'Running', 'Biking', 'Jumping rope'],
                    'Regular Sleep': ['waking up early', 'Going to bed on time'],
                    'Healthy Diet': ['not eating', 'Eating vegetables', 'Not eating salty food', 'Not eating sweets', 'Watching what I eat', 'Drinking water', 'Cutting back on sugar', 'Cutting back on salt', 'Watching my diet'],
                    'Exercise': ['exercising', 'Playing sports', 'Liking exercise'],
                    'Not Smoking': ['not smoking', 'Not smoking cigarettes', 'not cigarettes', 'quitting smoking', 'Staying away from cigarettes'],
                    'Smoking': ['enjoying smoking', 'smoking cigarettes','smoking','cigarettes'],
                    'Not Drinking': ['not drinking', 'no alcohol','not drinking alcohol', 'not drink', 'staying away from alcohol'],
                    'Drinking': ['drinking','alcohol', 'drinking alcohol', 'drinking alcohol'],
                    'Near Smokers': ['around people who smoke', 'family members smoke', 'secondhand smoke'],
                    'Near Pollution': ['smoke', 'pollution', 'dust', 'dust', 'construction'],
                    'Stress': ['stress', 'stress symptoms'],
                    'Working Hard': ['working hard','working']
                    }
            behaviour_key = ['Irregular Sleep', 'Unhealthy Diet', 'Sugary Drinks', 'Lack of Exercise', 'Regular Sleep', 'Healthy Diet', 'Exercise', 'Not Smoking', 'Smoking', 'Not Drinking', 'Drinking', 'Near Smokers', 'Near Pollution', 'Stress']
            behaviour_extion = []
            for i in range(len(data_raddit)):
                behaviour_list=[]
                for j in behaviour_key:
                    for lol in behaviour_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in data_raddit['token'].iloc[i]:
                            behaviour_list.append(j)
                        else:
                            behaviour_list.append('ไม่สามารถระบุได้')
                for j in behaviour_key:
                    for lol in behaviour_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in data_raddit['token_2g'].iloc[i]:
                            behaviour_list.append(j)
                        else:
                            behaviour_list.append('ไม่สามารถระบุได้')
                for j in behaviour_key:
                    for lol in behaviour_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in data_raddit['token_3g'].iloc[i]:
                            behaviour_list.append(j)
                        else:
                            behaviour_list.append('ไม่สามารถระบุได้')
                behaviour_list =list(set(behaviour_list))
                remove_negations(behaviour_list)
                if len(behaviour_list) >= 2:
                    behaviour_list.remove('ไม่สามารถระบุได้')
                behaviour_extion.append(behaviour_list)
            # วิธีการรักษา
            Treatment_value ={'Surgery':['ผ่าตัด','Surgery','ผ่า','operation', 'incision', 'surgery','บายพาส','สวนหัวใจ','ลดขนาดกระเพาะอาหาร','ตัดกระเพาะอาหาร'],
                            'Chemotherapy':['chemotherapy','chemo','คีโม','เคมี','herceptin','บำบัดรักษา','เคมีบำบัด'],
                            'Radiation Therapy':['radiation','ฉายแสง','เเร่','ฉายเเสง'],
                            'Targeted Therapy':['targeted','เป้า'],
                            "Diet control": ["ลดแป้ง","ลดน้ำตาล","ลดไขมัน", "กินผัก" ,"กินผลไม้" ,"กินธัญพืช", "เพิ่มใยอาหาร","ลดคอเลสเตอรอล","ลดปริมาณอาหาร","ลดปริมาณ"
                                        ,"ปรับเปลี่ยนพฤติกรรมการกิน", "เปลี่ยนการกิน","ลดโซเดียม", "ลดโพแทสเซียม",
                                        "reducestarch", "reducesugar", "reducefat", "eatvegetables", "eatfruits", "eatgrains", "increasedietary", "reducecholesterol", "Reducefood", "reduceintake"
                                        , "changeeat",  "reducesodium", "reducepotassium"],
                            "Drug use": ["อินซูลิน","ยาลดน้ำตาล","ยาลด","ยาลดความดัน", "ยาลดไขมัน","ยาลดความอยากอาหาร","กินยา",
                                        "insulin", "drugs", "diet drugs", "lowering", "suppressants", "medicine"],
                            "Self-care": ["ตรวจระดับน้ำตาล", "ดูแลเท้า","ลดเครียด","ลดความเครียด","พักผ่อนให้เพียงพอ","พักผ่อน",
                                        "bloodsugar", "care", "reducestress", "rest", "relax"],
                            }
            Treatment_key = ['Surgery','Chemotherapy','Radiation Therapy','Targeted Therapy','Diet control',"Drug use","Self-care"]
            Treatment_extion = []
            for i in range(len(data_raddit)):
                Treatment_list=[]
                for j in Treatment_key:
                    for lol in Treatment_value[j]:
                    # Use .iloc to access rows by numerical index
                        if lol in data_raddit['token'].iloc[i]:
                            Treatment_list.append(j)
                        elif lol in data_raddit['token_2g'].iloc[i]:
                            Treatment_list.append(j)
                        elif lol in data_raddit['token_3g'].iloc[i]:
                            Treatment_list.append(j)
                        else:
                            Treatment_list.append('ไม่สามารถระบุได้')
                Treatment_list =list(set(Treatment_list))
                if len(Treatment_list) >= 2:
                    Treatment_list.remove('ไม่สามารถระบุได้')
                Treatment_extion.append(Treatment_list)
            def detect_gender_other_en(comment):
                # คำที่บ่งบอกถึงเพศชาย
                male_keywords = ['he','him','dad', "father","stepfather","grandfather","great-grandfather","husband",
                "boyfriend","fiancé","son","stepson","grandson","great-grandson","brother","half-brother","stepbrother","uncle","great-uncle","nephew",
                "great-nephew","father-in-law","stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law","grandson-in-law","brother-in-law",
                "half-brother-in-law","stepbrother-in-law","uncle-in-law","great-uncle-in-law","nephew-in-law","great-nephew-in-law","godfather","stepfather",
                "godson","foster father","foster son","stepson","stepbrother","half-brother","great-uncle","great-nephew","first cousin","second cousin",
                "third cousin","father-in-law","stepfather-in-law","grandfather-in-law","son-in-law","stepson-in-law","grandson-in-law","brother-in-law",
                "half-brother-in-law","stepbrother-in-law","uncle-in-law","great-uncle-in-law","nephew-in-law",
                "great-nephew-in-law",'boy']
                # คำที่บ่งบอกถึงเพศหญิง
                female_keywords = ["mother",'mom',"stepmother",'girl',
                "grandmother","great-grandmother","wife","girlfriend","fiancée","daughter","stepdaughter",
                "granddaughter","great-granddaughter","sister","half-sister","stepsister","aunt","great-aunt","niece",
                "great-niece","mother-in-law","stepmother-in-law","grandmother-in-law","daughter-in-law","stepdaughter-in-law",
                "granddaughter-in-law","sister-in-law","half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law",
                "niece-in-law","great-niece-in-law","cousin-in-law","godmother","stepmother","goddaughter","foster mother","foster daughter","stepdaughter","stepsister",
                "half-sister","great-aunt","great-niece","mother-in-law","stepmother-in-law","grandmother-in-law","daughter-in-law","stepdaughter-in-law",
                "granddaughter-in-law","sister-in-law","half-sister-in-law","stepsister-in-law","aunt-in-law","great-aunt-in-law","niece-in-law","great-niece-in-law",'mommy','she','her']
                # ตรวจสอบคำในความคิดเห็น
                for keyword in male_keywords:
                    if keyword in comment :
                        return "เพศชาย"
                for keyword in female_keywords:
                    if keyword in comment:
                        return "เพศหญิง"
                return "ไม่สามารถระบุได้"
            k3=[]
            k4=[]
            for i in data_raddit['comments']:
                k3.append(detect_person_en(str(i)))
                if detect_person_en(str(i)) == "เล่าประสบการณ์ตัวเอง":
                    k4.append("ไม่สามารถระบุได้")
                elif detect_person_en(str(i)) == "เล่าประสบการณ์คนอื่น":
                    k4.append(detect_gender_other_en(str(i)))
                elif detect_person_en(str(i)) == "ไม่สามารถระบุได้":
                    k4.append(detect_gender_other_en(str(i)))
            #======================================================
            def analyze_symptoms_multi_level(data_reddit, ngram_cancer_symptoms):
                """วิเคราะห์ข้อมูล Reddit เพื่อหาอาการโดยใช้ list_token, token_2gam, และ token_3gam"""
                list_token_red = data_reddit['token']
                list_token_2gam_red = data_reddit['token_2g']
                list_token_3gam_red = data_reddit['token_3g']
                list_symptoms_en = []

                # วิเคราะห์ list_token
                for token in list_token_red:
                    for symptom, tokens in ngram_cancer_symptoms.items():
                        if token in tokens:
                            list_symptoms_en.append(symptom)

                # วิเคราะห์ token_2gam
                for token_pair in list_token_2gam_red:
                    for symptom, tokens in ngram_cancer_symptoms.items():
                        if token_pair in tokens:
                            list_symptoms_en.append(symptom)

                # วิเคราะห์ token_3gam
                for token_triple in list_token_3gam_red:
                    for symptom, tokens in ngram_cancer_symptoms.items():
                        if token_triple in tokens:
                            list_symptoms_en.append(symptom)

                unique_list_symptoms_en = list(OrderedDict.fromkeys(list_symptoms_en))
                print(unique_list_symptoms_en)

                if len(unique_list_symptoms_en) > 0:
                    return unique_list_symptoms_en
                else:
                    return ['No symptoms identified']
            #----------------------------------------------------------------------------------
            all_reddit_data['defind_Genden_with_nlp'] = new_colgenden_en
            all_reddit_data['defind_Genden_with_python'] = k4
            all_reddit_data['defind_exp_with_python'] = k3
            all_reddit_data['beha']= behaviour_extion
            all_reddit_data['Treatment'] = Treatment_extion
            comment_label = []
            for text in all_reddit_data['comments']:
                predicted_sentiment = predict_sentiment_thai(text,model_thai=model_eng, tokenizer_Autp_T=tokenizer_Autp_E, device=device)
                comment_label.append(predicted_sentiment)
            all_reddit_data['label'] = comment_label
            # all_reddit_data['use_ful'] = 'non'
            all_reddit_data['symptoms_colcan_en'] = all_reddit_data.apply(lambda row_3: analyze_symptoms_multi_level(row_3, cancer_symptoms_en), axis=1)
            label_symptoms_en=all_reddit_data['symptoms_colcan_en'].str.join(sep='*').str.get_dummies(sep='*')
            all_reddit_data = all_reddit_data.join(label_symptoms_en)
            all_reddit_data.to_csv('data_pre.csv', index=False, encoding='utf-8-sig')
#======================================================================================================================
        if soure == 'www.facebook.com':
            all_data=pd.read_csv('all_data_nameandsym.csv')
            data_use=pd.read_csv('data_pre.csv')
            name_cancarTH_filter = all_data['name_cancarTH_se'].dropna().to_list()
            name_symptomsTH_filter = all_data['Key_symptoms_TH'].dropna().to_list()
            data_column_sym = data_use.columns[18:]
            for i in data_column_sym:
                if i not in name_symptomsTH_filter:
                    data_use.drop(i, axis=1, inplace=True)
            rows_to_drop = data_use[~data_use['defind_cancer_with_nlp'].isin(name_cancarTH_filter)].index
            data_use.drop(rows_to_drop, axis=0, inplace=True)
        elif soure == 'www.reddit.com':
            #filter reddit
            all_data=pd.read_csv('all_data_nameandsym.csv')
            data_use=pd.read_csv('data_pre.csv')
            name_symptoms_filter_en = all_data['Key_symptoms_EN'].dropna().to_list()
            name_cancar_filter_en= all_data['cancer_names_en_se'].dropna().to_list()
            data_column_sym = data_use.columns[18:]
            for i in data_column_sym:
                if i not in name_symptoms_filter_en:
                    data_use.drop(i, axis=1, inplace=True)
            rows_to_drop = data_use[~data_use['defind_cancer_with_nlp'].isin(name_cancar_filter_en)].index
            data_use.drop(rows_to_drop, axis=0, inplace=True)
        data_use['sum_ch'] = sum_ch
        data_use.to_csv('data_pre_setting.csv', index=False, encoding='utf-8-sig')
#===========================================================================================================================
        sorue_sym_4 = pd.read_csv('soure_url.csv')
        sorue_chack = sorue_sym_4['url'][0]
        if sorue_chack == 'www.facebook.com': 
            _ = ['ยอดไลค์','จำนวนการตอบกลับ','จำนวนคำ']
            data =  data_use
            data_show = data_use.iloc[:,:5]
            def sort_data(column_name,how_sort):
                if column_name == 'ยอดไลค์':
                    data_show.sort_values('ยอดไลค์', inplace=True, ascending=how_sort)
                elif column_name == 'จำนวนการตอบกลับ':
                    data_show.sort_values('จำนวนการตอบกลับ', inplace=True, ascending=how_sort)
                elif column_name == 'จำนวนคำ':
                    data_show.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
                else:
                    pass
                return data_show
            sort_options = ['ยอดไลค์','จำนวนการตอบกลับ','จำนวนคำ']
            mylist_name_can = data['defind_cancer_with_nlp'].to_list()
            name_can  = list(dict.fromkeys(mylist_name_can))
            sym_list = data.columns
            symptoms_can = sym_list[16:]
            data_name_sym_have = pd.DataFrame()
            data_value_TH = pd.DataFrame(data={'name_cancarTH':name_can})
            data_symptoms_TH = pd.DataFrame(data={'Key_symptoms_TH':symptoms_can})
            list_100 = list(range(0,100))
            data_name_sym_have['index'] = list_100
            data_name_sym_have = data_name_sym_have.merge(data_value_TH.reset_index(), how='outer')
            data_name_sym_have = data_name_sym_have.merge(data_symptoms_TH.reset_index(), how='outer')
            data_name_sym_have.to_csv('data_name_sym_have.csv',encoding='utf-8-sig')
            #------word cloud---------#
            from pythainlp.tokenize import word_tokenize as to_th_k # เป็นตัวตัดคำของภาษาไทย
            from pythainlp.corpus import thai_stopwords # เป็นคลัง Stop Words ของภาษาไทย
            text_th= ''
            for row in data['comments']: # ให้ python อ่านข้อมูลรีวิวจากทุก row ใน columns 'content'
                text_th = text_th + row.lower() + ' ' # เก็บข้อมูลรีวิวของเราทั้งหมดเป็น String ในตัวแปร text

            wt_th = to_th_k(text_th, engine='newmm') # ตัดคำที่ได้จากตัวแปร text

            path_th = 'THSarabunNew-20240628T045147Z-001\THSarabunNew\THSarabunNew.ttf' # ตั้ง path ไปหา font ที่เราต้องการใช้แสดงผล
            wordcloud_th = WordCloud(font_path = path_th, # font ที่เราต้องการใช้ในการแสดงผล เราเลือกใช้ THSarabunNew
                                stopwords = thai_stopwords(), # stop words ที่ใช้ซึ่งจะโดนตัดออกและไม่แสดงบน words cloud
                                relative_scaling = 0.3,
                                min_font_size = 1,
                                background_color = "white",
                                width=620,
                                height=300,
                                max_words = 500, # จำนวนคำที่เราต้องการจะแสดงใน Word Cloud
                                colormap = 'plasma',
                                scale = 3,
                                font_step = 4,
                                collocations = False,
                                regexp = r"[ก-๙a-zA-Z']+", # Regular expression to split the input text into token
                                margin=2).generate(' '.join(wt_th)) # input คำที่เราตัดเข้าไปจากตัวแปร wt ในรูปแบบ string

            wordcloud_th.to_file("wordcloud.png")
            max_v = []
            min_v = []
            avg_v = []
            for i in range(len(data_show.columns)):
                if i >= 2 :
                    max_v.append(max(data_show[data_show.columns[i]].tolist()))
                    min_v.append(min(data_show[data_show.columns[i]].tolist()))
                    avg_v.append(np.mean(data_show[data_show.columns[i]].tolist()))
            descriptive = pd.DataFrame()
            descriptive[' '] = _
            descriptive['max'] = max_v
            descriptive['min'] = min_v
            descriptive['avg'] = avg_v
            descriptive.to_csv('data_desc.csv',encoding='utf-8-sig')
            sorted_data = sort_data('ยอดไลค์',False)
        elif sorue_chack == 'www.reddit.com':
            import pandas as pd
            from nltk.tokenize import word_tokenize as to_en
            from nltk.stem import WordNetLemmatizer
            data = all_reddit_data
            data_show = data_use.iloc[:,:3]
            _ = ['จำนวนคำ']
            def sort_data(column_name,how_sort):
                if column_name == 'จำนวนคำ':
                    data_show.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
                else:
                    pass
                return data_show
            sort_options = ['จำนวนคำ']
            mylist_name_can = data['defind_cancer_with_nlp'].to_list()
            name_can  = list(dict.fromkeys(mylist_name_can))
            
            symptoms_can = data.columns[16:]
            data_name_sym_have = pd.DataFrame()
            data_value_TH = pd.DataFrame(data={'cancer_names_en':name_can})
            data_symptoms_TH = pd.DataFrame(data={'Key_symptoms_EN':symptoms_can})
            list_100 = list(range(0,100))
            data_name_sym_have['index'] = list_100
            data_name_sym_have = data_name_sym_have.merge(data_value_TH.reset_index(), how='outer')
            data_name_sym_have = data_name_sym_have.merge(data_symptoms_TH.reset_index(), how='outer')
            data_name_sym_have.to_csv('data_name_sym_have.csv',encoding='utf-8-sig')
            import pandas as pd
            #------word cloud---------#
            # Combine all text into a single string
            text = ' '.join(data['comments'].str.lower())
            # Tokenize the text
            tokens = to_en(text)
            # Remove stop words and lemmatize
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            filtered_words = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
            # Create a word cloud
            wordcloud = WordCloud(
                stopwords=stop_words,
                background_color="white",
                width=620,
                height=300,
                max_words=500,
                colormap='plasma',
                scale=3,
                font_step=4,
                collocations=False,
                margin=2
            ).generate(' '.join(filtered_words))
            # Save the word cloud to a file
            wordcloud.to_file("wordcloud.png")
            max_v = []
            min_v = []
            avg_v = []
            for i in range(len(data_show.columns)):
                if i == 2:
                    max_v.append(max(data_show[data_show.columns[i]].tolist()))
                    min_v.append(min(data_show[data_show.columns[i]].tolist()))
                    avg_v.append(np.mean(data_show[data_show.columns[i]].tolist()))
            descriptive = pd.DataFrame()
            descriptive[' '] = _
            descriptive['max'] = max_v
            descriptive['min'] = min_v
            descriptive['avg'] = avg_v
            descriptive.to_csv('data_desc.csv',encoding='utf-8-sig')
            sorted_data = sort_data('จำนวนคำ',False)
        descriptive =pd.read_csv('data_desc.csv')
        descriptive = descriptive.iloc[:, 1:]
        tables_d = descriptive.to_html(classes='table table-striped', index=False)
        # เรียงลำดับข้อมูลตามค่าเริ่มต้น (like)
        sorted_data.to_csv('sorted_data.csv',encoding='utf-8-sig')
        tables = sorted_data.to_html(classes='table table-striped', index=False)
        count_user = len(set(sorted_data.iloc[:,0]))
        count_comment = len(sorted_data.iloc[:,1])
        number_of_rows = len(data)
        number_of_columns = len(data.columns)
            # for found in name_cancarTH_filter:
    x08 = pd.read_csv('ref_pas.csv')
    x18 = x08['ref'][0]
    x30 = x08['refy'][0]
    return render_template('output2.html', tables=[tables], sort_options=sort_options,skills=name_can,symptoms=symptoms_can
                    ,tables_descript=[tables_d],count_user=count_user,count_comment=count_comment,number_of_rows=number_of_rows,number_of_columns=number_of_columns
                    ,x=x18,y=x30)


@server.route('/page3.py',methods=["POST","GET"])
def index_2():
    x09 = pd.read_csv('ref_pas.csv')
    x09['refy'][0] = 10
    x30 = x09['refy'][0]
    x19 = x09['ref'][0]
    x09.to_csv('ref_pas.csv',index=False)
    try:
        data_for_dash = pd.read_csv('data_pre_setting.csv', encoding='utf-8-sig')
        data_for_dash.drop('sum_ch', axis=1, inplace=True)
        data_for_dash.to_csv('data_for_dash_01.csv',index=False,encoding='utf-8-sig')
        data_for_dash.to_csv('data_pre_setting.csv',index=False,encoding='utf-8-sig')
    except:
        data_for_dash = pd.read_csv('data_pre.csv', encoding='utf-8-sig')
        data_for_dash.to_csv('data_for_dash_01.csv',index=False,encoding='utf-8-sig')
        # ระบุพาธของ Python interpreter ใน virtual environment
    python_path = os.path.join(os.getcwd(), "env", "Scripts", "python.exe")

    # ตรวจสอบ environment variables ที่ถูกตั้งค่าใน virtual environment
    bot_sum = subprocess.Popen([python_path, "pd_to_text.py"], env=os.environ.copy())
    # bot_sum = subprocess.Popen("python pd_to_text.py",shell=True)
    bot_sum
    time.sleep(25)
    # proc1 = subprocess.Popen("python test_dash_for_bu.py",shell=True)
    # proc1 = subprocess.Popen("python Dashboard_for_sum_work_2025.py",shell=True)
    proc1 = subprocess.Popen([python_path, "Dashboard_for_sum_work_2025.py"], env=os.environ.copy())
    proc1
    time.sleep(25)
    time_run = pd.read_csv('time_run.csv')
    time_run.iloc[0,0] = 0
    time_run.to_csv('time_run.csv',index=False)
    return render_template('login2_2.html',x=x19,y=x30)

@server.route('/page3_2.py',methods=["POST","GET"])
def index_3456():
    time_run = pd.read_csv('time_run.csv')
    time_run.iloc[0,0] = 0
    time_run.to_csv('time_run.csv',index=False)
    return render_template('login2_2.html')

@server.route('/page2_3.py',methods=["POST","GET"])
def index_78():
    soure_b = pd.read_csv('soure_url.csv')
    soure = soure_b['url'][0]
    data = pd.read_csv("sorted_data.csv", encoding='utf-8-sig')
    data = data.iloc[:,1:]
    number_of_rows = len(data)
    number_of_columns = len(data.columns)
    count_user = len(set(data.iloc[:,0]))
    count_comment = len(data.iloc[:,1])
    if soure == 'www.facebook.com':
        def sort_data(column_name,how_sort):
            if column_name == 'ยอดไลค์':
                data.sort_values('ยอดไลค์', inplace=True, ascending=how_sort)
            elif column_name == 'จำนวนการตอบกลับ':
                data.sort_values('จำนวนการตอบกลับ', inplace=True, ascending=how_sort)
            elif column_name == 'จำนวนคำ':
                data.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
            else:
                pass
            return data
        sort_options = ['ยอดไลค์', 'จำนวนการตอบกลับ', 'จำนวนคำ']
        data_cancer= pd.read_csv('data_name_sym_have.csv')
        name_can = data_cancer['name_cancarTH'].dropna().to_list()
        symptoms_can = data_cancer['Key_symptoms_TH'].dropna().to_list()
    elif soure == 'www.reddit.com':
        def sort_data(column_name,how_sort):
            if column_name == 'จำนวนคำ':
                data.sort_values('จำนวนคำ', inplace=True, ascending=how_sort)
            else:
                pass
            return data
        sort_options = ['จำนวนคำ']
        data_cancer= pd.read_csv('data_name_sym_have.csv')
        name_can = data_cancer['cancer_names_en'].dropna().to_list()
        symptoms_can = data_cancer['Key_symptoms_EN'].dropna().to_list()
    descriptive=pd.read_csv('data_desc.csv',encoding='utf-8-sig')
    descriptive = descriptive.iloc[:, 1:]
    tables_d = descriptive.to_html(classes='table table-striped', index=False)
    sorted_data = sort_data('ความยาวของความคิดเห็น',False)
    tables = sorted_data.to_html(classes='table table-striped', index=False)
    x10 = pd.read_csv('ref_pas.csv')
    x20 = x10['ref'][0]
    x30 = x10['refy'][0] 
    time_run = pd.read_csv('time_run.csv')
    time_run.iloc[0,0] = 0
    time_run.to_csv('time_run.csv',index=False)
    return render_template('output2.html', tables=[tables], sort_options=sort_options,skills=name_can,symptoms=symptoms_can
                           ,tables_descript=[tables_d],number_of_rows=number_of_rows,number_of_columns=number_of_columns,count_user = count_user,count_comment=count_comment
                           ,x=x20,y=x30)

# @server.route('/page1_2', methods=['POST','GET'])
# def index5():
#   return render_template('input3.html')

# @server.route('/page1_3', methods=['POST','GET'])
# def index6():
#   return render_template('input4.html')

if __name__ == "__main__":
    server.run(host='0.0.0.0', port=5000)


