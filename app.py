import shutil
from create_video import VideoCreator
import streamlit as st
import os
import glob
import subprocess
import cv2
import PIL
from PIL import Image
from diffusion import DiffusionModel
from video_processor import VideoProcessor
from streamlit_option_menu import option_menu

def resize_images(image_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(image_directory, filename))
            img = img.resize((256, 256), PIL.Image.LANCZOS)
            img.save(os.path.join(image_directory, filename))

def copy_images(original_directory, new_directory):
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    for filename in os.listdir(original_directory):
        if filename.endswith('.png'):
            shutil.copy(os.path.join(original_directory, filename), new_directory)


def delete_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            os.remove(os.path.join(directory, filename))

def move_image_sequence(image_name):
    destination_dir = f'./Upscaler/datasets/input'
    source_file = f'./temp/enhance_video/{image_name}'

    shutil.move(source_file, destination_dir)

def delete_directory():
    dir_path = './results/'
    # check if the directory exists
    if os.path.exists(dir_path):
    # remove all folders and files inside the directory
        shutil.rmtree(dir_path)
    # create the directory again
    os.makedirs(dir_path)
 
def renaming():
    #renaming
    directory = '/test_video'

    # find all files in the directory that contain '_HAT_GAN_Real_SRx4'
    for filename in glob.glob(os.path.join(directory, '*_HAT_GAN_Real_SRx4*')):
        # construct new filename by replacing '_HAT_GAN_Real_SRx4' with ''
        new_filename = filename.replace('_HAT_GAN_Real_SRx4', '')
        # rename the file
        os.rename(filename, new_filename)
    
# Function to save the uploaded file
def save_uploaded_file(uploadedfile):
    file_path = os.path.join("temp", uploadedfile.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def move_image(image_name):
    destination_dir = f'./Upscaler/datasets/input'
    source_file = f'./temp/enhance/{image_name}'


    for img_file in glob.glob(destination_dir + '/*.png'):
        os.remove(img_file)

    shutil.move(source_file, destination_dir)

def clean_directory():
    directory = "./results"

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

def upscale_image():
    command = ["python3", "HAT/test.py", "-opt", "Upscaler/options/test/HAT_GAN_Real_SRx4.yml"]
    result = subprocess.run(command, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)


def upscale_video(image_directory):


    dst_image_path = f'./test_video/'

    for filename in os.listdir(image_directory):
        if filename.endswith('.png'):
            img = Image.open(f'{image_directory}/{filename}')
            img = img.resize((512, 512),PIL.Image.LANCZOS)
            img.save(os.path.join(dst_image_path, filename))
   
def delete_png_files(folder_path):
    search_pattern = os.path.join(folder_path, '*.png')
    png_files = glob.glob(search_pattern)
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def cleanup_video():
    delete_png_files("./dataset/video/hr_256/")
    delete_png_files("./dataset/video/sr_16_256/")
    delete_png_files("./temp/enhance_video/")
    delete_png_files("./temp/image_sequence/")
    delete_png_files("./test_video/")
    shutil.rmtree("./experiments_train")         
        
with st.sidebar:
    selected = option_menu("Main Menu", ["Image", 'Video'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    

if selected == "Image":
    st.title('Image Processing App')
    # Upload the file
    uploaded_file = st.file_uploader("Upload an png image", type=['png'], label_visibility="hidden")
    progress_text = "Operation in progress. Please wait."

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)
        
        # Open the image file
        img = Image.open(file_path)

        # Resize the image
        img = img.resize((256, 256))

        # Save the image as a PNG file
        img.save('./dataset/test/hr_256/00001.png')
        img.save('./dataset/test/sr_16_256/00001.png')

    if st.button('Execute'):
        st.button
        progress_bar = st.progress(30, text=progress_text)

        model = DiffusionModel(image_name = uploaded_file.name)
        model.run()
        progress_bar.progress(50, text=progress_text)
        move_image(uploaded_file.name)
        clean_directory()
        progress_bar.progress(70, text=progress_text)

        upscale_image()
        print("Finish Infering")
        progress_bar.progress(100, text=progress_text)

        name, extension = os.path.splitext(uploaded_file.name)
        image_name = f'{name}_HAT_GAN_Real_SRx4.png'
        image_path = f'./results/HAT_GAN_Real_SRx4/visualization/custom/{image_name}'
        progress_bar.empty()
        st.image(image_path, caption='Sunrise by the mountains')
        
        
        print("Finish Display Result")


if selected == "Video":
    st.title('Video Processing App')

    if st.button("Convert Images to Video"):
        progress_bar = st.progress(0, text=progress_text)

        processor = VideoProcessor(path="underwater.mp4", threshold=30.0, step=1, save="./temp/image_sequence/")
        processor.process_video()
        print("resizing images")

        resize_images('./temp/image_sequence/') 
        print("moving images")
        progress_bar = st.progress(15, text=progress_text)

        copy_images('./temp/image_sequence/','./dataset/video/hr_256/')
        copy_images('./temp/image_sequence/','./dataset/video/sr_16_256/')    
        print("enhance images")
        progress_bar.progress(40, text=progress_text)
        model = DiffusionModel(mode=2)
        model.run()
        progress_bar.progress(70, text=progress_text)

        print("upscaling")
        upscale_video("./temp/enhance_video/")
        video_creator = VideoCreator(img_dir="./test_video/", video="underwater_enhance.mp4", fps=30)
        video_creator.create_video() 
        progress_bar.progress(90, text=progress_text)
 
        cleanup_video()     
        progress_bar.progress(100, text=progress_text)
 
        st.success("Video created successfully!")
