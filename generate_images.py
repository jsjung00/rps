import pickle 
import cv2 
import numpy as np 
import time 
import matplotlib.pyplot as plt
from PIL import Image  
def generate_images(sample_size, save_file='new_saved_images.pkl'):
    images = {'rock':[],'paper':[], 'scissors':[]}
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    height = int(cap.get(4))
    box_width = 200
    category = None 
    capture = False 
    capture_count = 0
    while True: 
        ret,frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (width - width//10 - box_width, height//2 - box_width//2), (width - width//10, height//2 + box_width//2), (0, 250, 150), 2)
        cv2.imshow('frame', frame)
        if (capture):
            if capture_count >= sample_size:
                cv2.putText(frame, f"Finished capture", (50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
                capture_count = 0
                capture = False 
                category = None 
            else: 
                print(f'category {category}')
                #print(f'frame shape {frame.shape}')
                cv2.waitKey(1)
                width_l = width - width//10 - box_width
                width_r = width - width//10
                height_t = height//2 - box_width//2
                height_b = height//2 + box_width//2
                #print(f'frame[{width_l}:{width_r}, {height_t}:{height_b}]')
                #box = frame[width - width//10 - box_width:width - width//10, height//2 - box_width//2 : height//2 + box_width//2, :]
                box = frame[height_t:height_b, width_l:width_r]
                images[category].append(np.array(box))
                capture_count += 1 
        else:
            key = cv2.waitKey(100)
            capture=True 
            print(f'key {key}')
            if key == ord("r"):
                category="rock"
            elif key == ord("s"):
                category = "scissors"
            elif key == ord("p"):
                category = "paper"
            elif key == ord("t"):
                #save and terminate
                with open(save_file, 'wb') as f:
                    pickle.dump(images, f)
                return 
            elif key == ord("b"):
                #terminate without saving
                return 
            else:
                capture=False 
#generate_images(10, save_file="test_images.pkl")

def load_pickle():
    with open('new_saved_images.pkl', "rb") as f:
        images = pickle.load(f)  
    fig = plt.figure(figsize=(10, 10))
    rows = 3
    cols = 10
    keys = list(images.keys())
    _, axs = plt.subplots(rows, cols, figsize=(12, 12))
    for i in range(0, len(keys)):
        key = keys[i]
        idx = np.round(np.linspace(0, len(images[key]) - 1, cols)).astype(int)
        type_images = np.array(images[key])[idx]
        for j in range(0, cols):
            axs[i, j].imshow(type_images[j])
    plt.show()
#load_pickle()



def create_imagefolder(rt, save_file):
    with open(save_file, "rb") as f:
        images = pickle.load(f)  
    keys = list(images.keys())
    for key in keys:
        cat_images = images[key]
        for e, im in enumerate(cat_images):
            im = im.astype(np.uint8)
            pil_image = Image.fromarray(im).convert('RGB')
            pil_image.save(f'{rt}\\test\\{key}\\{key}{e+1}.png')
create_imagefolder('C:\\Users\\justi\\Desktop\\Everything\\Code\\rps', "test_images.pkl")







    