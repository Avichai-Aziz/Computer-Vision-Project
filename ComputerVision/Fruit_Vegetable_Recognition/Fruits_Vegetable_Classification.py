import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import os

model = load_model('FV.h5')

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'broccoli', 5: 'cabbage', 6: 'capsicum', 7: 'carrot', 8: 'cauliflower', 9: 'chilli pepper', 10: 'corn', 11: 'cucumber', 12: 'eggplant', 13: 'garlic', 14: 'ginger', 15: 'grapes', 16: 'jalepeno', 17: 'kiwi', 18: 'lemon',
          19: 'lettuce', 20: 'mango', 21: 'onion', 22: 'orange', 23: 'papaya', 24: 'paprika', 25: 'pear', 26: 'peas', 27: 'pineapple', 28: 'pomegranate', 29: 'potato', 30: 'raddish', 31: 'soy beans', 32: 'spinach', 33: 'sweetcorn', 34: 'sweetpotato', 35: 'tomato'}


fruits = ['Apple','Banana','Bello Pepper','Chilli Pepper','Grapes','Jalepeno','Kiwi','Lemon','Mango','Orange','Paprika','Pear','Pineapple','Pomegranate', 'Papaya']
vegetables = ['Beetroot','Cabbage','Capsicum','Carrot','Cauliflower','Corn','Cucumber','Eggplant','Ginger','Lettuce','Onion','Peas','Potato','Raddish','Soy Beans','Spinach','Sweetcorn','Sweetpotato','Tomato', 'Broccoli']

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result= processed_img(save_image_path)
            print(result)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            st.success("**Predicted : "+result+'**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**'+cal+'(100 grams)**')
run()