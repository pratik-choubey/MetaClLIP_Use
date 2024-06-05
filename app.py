from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import torch
from pathlib import Path
import glob
import streamlit as st
import os
 

#load a pretrained model
model = CLIPModel.from_pretrained("local_model")
processor = CLIPProcessor.from_pretrained("local_model")
tokenizer = AutoTokenizer.from_pretrained("local_model")
# Function to encode text
def encode_text(text):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_embedding = model.get_text_features(**inputs)
    return text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
 
# Function to get image embeddings
def encode_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_embedding = model.get_image_features(**inputs)
    return image_embedding/ image_embedding.norm(p=2, dim=-1, keepdim=True)

 
#Function to get similarity
def Similarity(text_emb,img_emb):
    similarities = []
    for i in range(len(img_emb)):
       sim  = torch.matmul(text_emb, img_emb[i].t())
       similarities.append(sim)
    return similarities


 
#Run
def search(query,img_names,k=1):
    text_emb = encode_text(query)
    img_emb = [encode_image(img_path) for img_path in img_names]
    similarity_scores = list(zip(img_names, Similarity(text_emb,img_emb)))
    # Sort images based on similarity scores
    sorted_images = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # print(sorted_images)
    matchd_img = []
    for idx, (img_names, similarity) in enumerate(sorted_images):
        if idx <k:
            matchd_img.append(Image.open(img_names))
    return matchd_img


# q = input("Enter your query here....")
if __name__ == "__main__":
    st.title("MetaCLIP-Text to Image search")
    st.write(
    """

    Hi 👋, I'm **:red[Pratik Chobey]**, and welcome to my **:green[Image Retrival application]**! :rocket: This program makes use of Hugging Face **:orange[Transformers]** model(Metaclip by facebook), 
    which can be used for image search and for zero-shot image classification.  ✨

    """
    )
    #Images input
    # img_names = st.file_uploader("Upload Multiple Image:",accept_multiple_files=True)
    input_data_path = st.text_input("Please provide data folder name....")
    # data_path = BASE_DIR/input_data_path
    data_path = input_data_path + "/"
    images = list(glob.glob(f'{data_path}*.JPEG')) #please specify the format of images
    thumbnails_container = st.container()
    thumbnails_container.image([Image.open(image).resize((100,100)) for image in images],width=100)
    q = st.text_input("Enter you query here....")
    k = int(st.number_input("Please enter number of similar images you want...",min_value=1,step=1,value=4))
    if st.button('RUN'):
        result = search(q, images,k)
        thumbnails_container_output = st.container()
        thumbnails_container_output.image([image.resize((100,100)) for image in result], width=100)
