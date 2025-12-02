import torch
import cv2 as cv
from model import RSLRmodel

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RSLRmodel().to(device)
model.load_state_dict(torch.load(f="./models/rslr_model_best.pth"))

# Setup camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    ret, frame = cap.read()
    resized_frame = cv.resize(
        src=frame, 
        dsize=(64, 64)
    )
    tensor_frame = torch.Tensor(resized_frame).permute(2, 0, 1).unsqueeze(dim=0).to(device)

    res = model(tensor_frame).argmax(dim=1)
    print(res)
    # if not ret:
    #     print("No frame got. Check camera status.")
    
    

    # cv.imshow("Camera", resized_frame)
    # if cv.waitKey(1) == ord("s"):
    #     break

cap.release()
cv.destroyAllWindows()