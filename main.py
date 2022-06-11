import pickle 
import cv2
import time  
from datetime import datetime 
from model import Shift_Net
from model import par
import torch 
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))
box_width = 200
start_time = datetime.now()
time_scale=2
class_to_idx = {0:'paper',  1:'rock', 2:'scissors'}
text = {0: "Rock...",1:"Paper...", 2:"Scissors...",3:"Shoot!"}

first_picture_path = "./data/paper/paper1.png"
first_picture = Image.open(first_picture_path)
transform = transforms.Compose([
    transforms.ToTensor()
])
first_picture_tensor = transform(first_picture)
print(f'First picture tensor {first_picture_tensor}') 
pars = par()
pars.kernel_size=[5,5]
pars.inp_dim = first_picture_tensor.size()

MODEL_PATH = "./saved_models/model.pth"
cnn_model = Shift_Net(pars).to(device)
cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
cnn_model.eval()

with open('paper_test.pkl', "rb") as f:
    images = pickle.load(f)  
    cat_images = images['paper']
    for e, im in enumerate(cat_images):
        im = im.astype(np.uint8)
        pil_image = Image.fromarray(im).convert('RGB')
        plt.imshow(pil_image)
        plt.show()
        box_t = transform(im) 
        box_t = torch.unsqueeze(box_t, dim=0)
        
        print(f'box_t shape {box_t.size()}')
        print(box_t)
        output = cnn_model(box_t)
        pred = torch.max(output,1)[1]
        print(f'output: {output}, prediction: {pred}')
        break 


while(False): 
    count_down_sec = (time_scale*(datetime.now() - start_time)).seconds
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"{text[count_down_sec % 4]}", (50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
    width_l = width - width//10 - box_width
    width_r = width - width//10
    height_t = height//2 - box_width//2
    height_b = height//2 + box_width//2
    cv2.rectangle(frame, (width_l, height_t), (width_r, height_b), (0, 250, 150), 2)
    cv2.imshow('frame', frame)
    if count_down_sec == 3:
        box = frame[height_t:height_b, width_l:width_r]
        box_t = transform(box) 
        box_t = torch.unsqueeze(box_t, dim=0)
        
        print(f'box_t shape {box_t.size()}')
        print(box_t)
        output = cnn_model(box_t)
        pred = torch.max(output,1)[1]
        print(f'output: {output}, prediction: {pred}')
    if count_down_sec == 4:
        time.sleep(1)
        start_time = datetime.now()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
#cap.release()
#cv2.destoryAllWindows()

