import io
import shutil
from Prompter.utils import load_img, saveImage
from create_video import VideoCreator
import streamlit as st
import os
import glob
import subprocess
import cv2
import PIL
import time
from PIL import Image
from diffusion import DiffusionModel
from video_processor import VideoProcessor
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from prompter import InstructIRProcessor


def resize_images(image_directory,w=256,h=256):
    for filename in os.listdir(image_directory):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(image_directory, filename))
            ow, oh = img.size
            img = img.resize((w, h), PIL.Image.LANCZOS)
            img.save(os.path.join(image_directory, filename))
    return ow, oh

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
    
def move_image_path(image_path):
    destination_dir = f'./Upscaler/datasets/input'
    source_file = image_path
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
    command = ["python3", "Upscaler/test.py", "-opt", "Upscaler/options/test/HAT_GAN_Real_SRx4.yml"]
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
    for item in os.listdir('./experiments_train/'):
        item_path = os.path.join('./experiments_train/', item)  # Full path to the item
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the direc
    
def cleanup_image():
    delete_png_files("./dataset/test/hr_256/")
    delete_png_files("./dataset/test/sr_16_256/")
    delete_png_files("./temp/enhance_video/")
    delete_png_files("./temp/image_sequence/")
    delete_png_files("./test_video/")
    #delete_png_files("./temp/")
    delete_png_files("./Upscaler/datasets/input/")
    for item in os.listdir('./experiments_train/'):
        item_path = os.path.join('./experiments_train/', item)  # Full path to the item
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the direc
    shutil.rmtree("./results/HAT_GAN_Real_SRx4")         

def enhance_image(width=512,height=512):
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
    image = Image.open(image_path)
    new_image = image.resize((width, height), PIL.Image.LANCZOS)
    cleanup_image()
    print("Finish Display Result") 

    return new_image


IMAGE_TO_URL = {
    "sample_image_1": "./dolphino.png/",
    "sample_image_2": "./dolphin.png/",
}            

if 'CURRENT_IMAGE' not in st.session_state:
    st.session_state['CURRENT_IMAGE'] = None

#CURRENT_IMAGE = None

@st.cache_data
def load_model():
    # Code to load your model goes here
    # For example, loading a machine learning model
    processor = InstructIRProcessor()
    return processor , True

processor, new_upload = load_model()

with st.sidebar:
    selected = option_menu("Main Menu", ["Picture", "Image", 'Video', 'Comparator'], 
        icons=['camera','file-image', 'file-play', 'gear'], menu_icon="cast", default_index=1)
if selected == "Comparator":
    with st.form(key="Streamlit Image Comparison"):
        col1, col2 = st.columns([3, 1])
        with col1:
            img1_url = st.text_input("Image one URL:", value=IMAGE_TO_URL["sample_image_1"])
        with col2:
            img1_text = st.text_input("Image one text:", value="ORIGINAL")

        # image two inputs
        col1, col2 = st.columns([3, 1])
        with col1:
            img2_url = st.text_input("Image two URL:", value=IMAGE_TO_URL["sample_image_2"])
        with col2:
            img2_text = st.text_input("Image two text:", value="ClearWaters")

       

        # centered submit button
        col1, col2, col3 = st.columns([6, 4, 6])
        with col2:
            submit = st.form_submit_button("Update Render 🔥")

    # render image-comparison
    static_component = image_comparison(
        img1=img1_url,
        img2=img2_url,
        label1=img1_text,
        label2=img2_text,
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

if selected == "Picture":
    st.title("Realtime Capture")

    progress_text = "Operation in progress. Please wait."
    photo_name = "photo.png"
    photo = st.camera_input("Take a picture")
    if photo is not None:
        image = Image.open(photo)
        st.image(image, caption='Original Image')
        st.session_state['CURRENT_IMAGE'] = image
            
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Example Prompt: Enhance my undewater image"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        CURRENT_IMAGE = st.session_state['CURRENT_IMAGE'] 
        
        if "enhance" in prompt.lower():
            print("Do enhance Task")
            CURRENT_IMAGE = CURRENT_IMAGE.resize((256, 256))
            CURRENT_IMAGE.save('./dataset/test/hr_256/00001.png')
            CURRENT_IMAGE.save('./dataset/test/sr_16_256/00001.png')
            CURRENT_IMAGE = enhance_image(width=512,height=512)
            st.chat_message("assistant").image(CURRENT_IMAGE, caption="Result", use_column_width=True)

        else:
            print("Do other image task")
            progress_bar = st.progress(50, text="Operation in progress. Please wait.")
            CURRENT_IMAGE.save('temp/prompter/prompt.png')
            CURRENT_IMAGE = load_img('temp/prompter/prompt.png')
            CURRENT_IMAGE = processor.process_img(CURRENT_IMAGE,prompt)
            out_path = "temp/prompter/out.png"
            saveImage(out_path, CURRENT_IMAGE)
     
            CURRENT_IMAGE = Image.open(out_path)
            CURRENT_IMAGE = CURRENT_IMAGE.resize((512, 512), PIL.Image.LANCZOS)
            progress_bar.progress(100, text=progress_text)
            progress_bar.empty()
            
            st.chat_message("assistant").image(CURRENT_IMAGE, caption='Result', use_column_width=True)

if selected == "Image":
    uploaded_image = None
    st.title('Image Processing App')
    # Upload the file
    uploaded_file = st.file_uploader("Upload an png image", type=['png'], label_visibility="hidden")
    progress_text = "Operation in progress. Please wait."
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)
        # Open the image file
        uploaded_image = Image.open(file_path)
        width, height = uploaded_image.size      
        st.image(uploaded_image, caption='Original Image')
        
    if st.button('Use Uploaded Image'):              
        st.session_state['CURRENT_IMAGE'] = uploaded_image
        st.toast('New Image is loaded into A.I ')
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Example Prompt: Enhance my undewater image"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if "enhance" in prompt.lower():
            print("Do enhance Task")
            CURRENT_IMAGE = st.session_state['CURRENT_IMAGE'] 
            CURRENT_IMAGE = CURRENT_IMAGE.resize((256, 256))
            CURRENT_IMAGE.save('./dataset/test/hr_256/00001.png')
            CURRENT_IMAGE.save('./dataset/test/sr_16_256/00001.png')
            CURRENT_IMAGE = enhance_image(width=width,height=height)
            st.chat_message("assistant").image(CURRENT_IMAGE, caption="Result", use_column_width=False)
            st.session_state['CURRENT_IMAGE'] = CURRENT_IMAGE

        else:
            CURRENT_IMAGE = st.session_state['CURRENT_IMAGE'] 
            print("Do other image task")
            progress_bar = st.progress(50, text="Operation in progress. Please wait.")
            CURRENT_IMAGE.save('temp/prompter/prompt.png')
            CURRENT_IMAGE = load_img('temp/prompter/prompt.png')
            CURRENT_IMAGE = processor.process_img(CURRENT_IMAGE,prompt)
            out_path = "temp/prompter/out.png"
            saveImage(out_path, CURRENT_IMAGE)
     
            CURRENT_IMAGE = Image.open(out_path)
            #CURRENT_IMAGE = CURRENT_IMAGE.resize((width, height), PIL.Image.LANCZOS)
            progress_bar.progress(100, text=progress_text)
            progress_bar.empty()
            
            st.chat_message("assistant").image(CURRENT_IMAGE, caption='Result', use_column_width=False)
            st.session_state['CURRENT_IMAGE'] = CURRENT_IMAGE



if selected == "Video":
    progress_text = "Operation in progress. Please wait."

    st.title('Video Processing App')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    temporary_location = False

    if uploaded_file is not None:
        g = io.BytesIO(uploaded_file.read()) 
        temporary_location = f'./test_video/{uploaded_file.name}'

        with open(temporary_location, 'wb') as out:  
            out.write(g.read()) 
        out.close()
        
    if st.button("Convert Images to Video"):
        
        progress_bar = st.progress(15, text=progress_text)

        processor = VideoProcessor(path=temporary_location, threshold=30.0, step=1, save="./temp/image_sequence/")
        processor.process_video()
        print("resizing images")

        vidw, vidh = resize_images('./temp/image_sequence/',256,256) 
        print("moving images")
        progress_bar.progress(25, text=progress_text)

        copy_images('./temp/image_sequence/','./dataset/video/hr_256/')
        copy_images('./temp/image_sequence/','./dataset/video/sr_16_256/')    
        print("enhance images")
        progress_bar.progress(40, text=progress_text)
        model = DiffusionModel(mode=2)
        model.run()
        progress_bar.progress(70, text=progress_text)

        print("upscaling")
        upscale_video("./temp/enhance_video/")
        resize_images('./test_video/',256,256)

        video_creator = VideoCreator(img_dir="./test_video/", video=f'./video/{uploaded_file.name}', fps=30, fourcc="XVID")
        video_creator.create_video() 
        progress_bar.progress(90, text=progress_text)
        time.sleep(5)
        progress_bar.progress(100, text=progress_text)
        time.sleep(5)
        st.success("Video created successfully!")
        #progress_bar.empty()
        video_name = f'./video/{uploaded_file.name}'
        video_file = open(uploaded_file.name, 'rb')
        video_bytes = video_file.read()
        video_name2 = f'./{uploaded_file.name}'
        video_file2 = open(video_name2, 'rb')
        video_bytes2 = video_file2.read()

        #st.video(video_bytes)
        col1, mid, col2 = st.columns([40,1,50])
        with col1:
            st.video(video_bytes2)
        with col2:
            st.video(video_bytes)
        cleanup_video()   
          

