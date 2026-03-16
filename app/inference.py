
from app.sigNet_pytorch_implementation import SiameseNetwork
import torch
import cv2
import numpy as np
from app.preprocessing import preprocessing_images_for_prediction, calculate_std_images
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = SiameseNetwork()
    model.load_state_dict(torch.load("model/siganet_trained_state_dic.pth"))
    model.eval()
    model = model.to(device)

    return model


def predict(content1, content2, model):
    np_array1 = np.frombuffer(content1, np.int8)
    np_array2 = np.frombuffer(content2, np.int8)

    img1 = cv2.imdecode(np_array1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np_array2, cv2.IMREAD_GRAYSCALE)



    std = calculate_std_images([img1, img2])


    with torch.no_grad():

        std = calculate_std_images([img1, img2])
        img1_processed = preprocessing_images_for_prediction(img1, std)
        img2_processed = preprocessing_images_for_prediction(img2, std)

        img1_processed = img1_processed.to(device)
        img2_processed = img2_processed.to(device)
        
        # emb1, emb2 = model(img1_processed,img2_processed)
        emb1, emb2 = model(img1_processed, img2_processed)
        # emb2 = model(img2_processed)
        distance = F.pairwise_distance(emb1,emb2)

    result=""
    if distance > 0.27:
        result = {f"mismatch"}
    else:
        result = {f"Match"}

    confidence = (torch.exp(-distance)*100).item()
    confidence_result = f"{confidence:.1f}" + "%"

    return {
        "result": result,
        "confidence": confidence_result
        }