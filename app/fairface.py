from pathlib import Path
import torchvision
from torchvision import transforms
import torch
import numpy as np


def load_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_folder = str(Path(__file__).parent.absolute())

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = torch.nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load(base_folder+'/res34_fair_align_multi_7_20190809.pt', 
        map_location=device))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = torch.nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load(base_folder+'/fairface_alldata_4race_20191111.pt', 
        map_location=device))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    return model_fair_4, model_fair_7


model_fair_4, model_fair_7 = load_models()


def predict_race4(face):
    """
    Edited from UCLA FairFace project:
    https://github.com/dchen236/FairFace

    :face:
    :returns: race4
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = trans(face)
    image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
    image = image.to(device)

    # fair 4 class
    outputs = model_fair_4(image)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.squeeze(outputs)

    race_outputs = outputs[:4]
    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    race_pred = np.argmax(race_score)

    race_scores_fair_4 = race_score
    race_preds_fair_4 = race_pred

    return race_preds_fair_4

    
def predict_race7(face):
    """
    Edited from UCLA FairFace project:
    https://github.com/dchen236/FairFace

    :face:
    :returns: [race7, age, gender]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = trans(face)
    image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
    image = image.to(device)

    # fair
    outputs = model_fair_7(image)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.squeeze(outputs)

    race_outputs = outputs[:7]
    gender_outputs = outputs[7:9]
    age_outputs = outputs[9:18]

    race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
    gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
    age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

    race_pred = np.argmax(race_score)
    gender_pred = np.argmax(gender_score)
    age_pred = np.argmax(age_score)

    race_scores_fair = race_score
    gender_scores_fair = gender_score
    age_scores_fair = age_score

    race_preds_fair = race_pred
    gender_preds_fair = gender_pred
    age_preds_fair = age_pred

    return race_preds_fair, age_preds_fair, gender_preds_fair