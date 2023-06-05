import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Chargement des modeles. Ecr

UNET = tf.keras.models.load_model('data/Models/UNET_masks.h5')
UNET.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalAccuracy'])

VGG16_6 = tf.keras.models.load_model("data/Models/VGG16_pt_6cat_4layers.h5", compile=False)
VGG16_6.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

VGG16_8 = tf.keras.models.load_model("data/Models/VGG16_pt_8cat_wh_aug_masked.h5", compile=False)
VGG16_8.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

VGG16_11 = tf.keras.models.load_model("data/Models/VGG16_pt_11cat_wh_aug_masked.h5", compile=False)
VGG16_11.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

VGG19_6 = tf.keras.models.load_model("data/Models/VGG19_pt_6cat_4layers.h5", compile=False)
VGG19_6.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

VGG19_11 = tf.keras.models.load_model("data/Models/VGG19_pt_11cat_4layers.h5", compile=False)
VGG19_11.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Chargement des classes 

class_names_6 = ['basophil', 'eosinophil', 'erythroblast','lymphocyte', 'monocyte', 'platelet']
class_names_8 = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
class_names_11 = ['basophil', 'BNE', 'eosinophil', 'erythroblast', 'lymphocyte', 'MMY', 'monocyte','MY', 'platelet', 'PMY', 'SNE']

# Page classification

st.markdown("Classification")
st.sidebar.header("Options")

uploaded_file = st.file_uploader('Load image', type=['png', 'jpg', 'jpeg', 'tiff'])

mask_apply = st.checkbox("Appliquer un masque", value=False)
prediction = st.checkbox("Prédire", value=False)

col1, col2, col3 = st.columns(3)
image_shape = (128, 128)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_shape)

    with col1:
        st.text("Original image")
        st.image(img)

    if mask_apply:
        # Expand dim of img
        img_exp = np.expand_dims(img,0)
        # Get UNET mask
        pred = UNET.predict(img_exp, verbose=0)
        mask = tf.argmax(pred, axis=-1)
        mask = mask[..., tf.newaxis]
        mask = np.uint8((mask[0].numpy()))
        # Display mask
        with col2:
            st.text("Mask")
            st.image(mask * 255)
        # Apply mask on img
        masked = cv2.bitwise_and(img, img, mask=mask)
        # Display masked image
        with col3:
            st.text("Masked image")
            st.image(masked)
            
    if prediction:
        options= st.multiselect('Choice model:',['VGG16_6', 'VGG16_8', 'VGG16_11','VGG19_6', 'VGG19_11'])
        result_table = []
        masked_norm = masked/255
        img_tens = tf.constant(masked_norm)
        img_tens = tf.expand_dims(img_tens, 0)

        if 'VGG16_6' in options:
            predictions = VGG16_6.predict(img_tens)
            pred = np.argmax(predictions, axis=1)
            pred_label = class_names_6[pred[0]]
            pred_score = predictions[0][pred[0]]
            result_table.append(['VGG16_6', pred_label, round(pred_score * 100, 2)])
        
        if 'VGG16_8' in options:
            predictions = VGG16_8.predict(img_tens)
            pred = np.argmax(predictions, axis=1)
            pred_label = class_names_8[pred[0]]
            pred_score = predictions[0][pred[0]]
            result_table.append(['VGG16_8', pred_label, round(pred_score * 100, 2)])
            
        if 'VGG16_11' in options:
            predictions = VGG16_11.predict(img_tens)
            pred = np.argmax(predictions, axis=1)
            pred_label = class_names_11[pred[0]]
            pred_score = predictions[0][pred[0]]
            result_table.append(['VGG16_11', pred_label, round(pred_score * 100, 2)])

        if 'VGG19_6' in options:
            predictions = VGG19_6.predict(img_tens)
            pred = np.argmax(predictions, axis=1)
            pred_label = class_names_6[pred[0]]
            pred_score = predictions[0][pred[0]]
            result_table.append(['VGG19_6', pred_label, round(pred_score * 100, 2)])
        
        if 'VGG19_11' in options:
            predictions = VGG19_11.predict(img_tens)
            pred = np.argmax(predictions, axis=1)
            pred_label = class_names_11[pred[0]]
            pred_score = predictions[0][pred[0]]
            result_table.append(['VGG19_11', pred_label, round(pred_score * 100, 2)])


        result_table = pd.DataFrame(result_table, columns=['Modèle', 'Prédiction', 'Score'])
            
            # Afficher tableau
          
        st.markdown("Résultats")
        st.dataframe(result_table)
