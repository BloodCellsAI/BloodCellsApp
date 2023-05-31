import streamlit as st
import numpy as np
import cv2
import tensorflow as tf


st.set_page_config(page_title="Classification", page_icon="📱")

UNET = tf.keras.models.load_model('data/Models/UNET_masks.h5')
VGG16 = tf.keras.models.load_model("data/Models/VGG16_pt_6cat_4layers.h5", compile=False)
VGG16.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
class_names = ['basophil', 'eosinophil', 'erythroblast','lymphocyte', 'monocyte', 'platelet']

st.markdown("# Classification")
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
        # Normalisation de l'image masquée
        masked_norm = masked/255
        # Conversion en tensor
        img_tens = tf.constant(masked_norm)
        # Mise en batch
        img_tens = tf.expand_dims(img_tens, 0)
        # Prédiction
        predictions = VGG16.predict(img_tens)
        pred = np.argmax(predictions, axis=1)
        pred_label = class_names[pred[0]]
        pred_score = predictions[0][pred[0]]
        # Affichage de la prédiction
        st.text(f"Cette image est un {pred_label} pour un score de {round(pred_score * 100, 2)} %.")



        


