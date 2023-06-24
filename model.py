import torch
import torchvision.transforms as transforms
from PIL import Image
from firebase import storage, database
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.to(device)

def PetImagePrediction(filepath):
    # Deschide și convertește imaginile în format RGB
    image_array = Image.open(filepath).convert("RGB")
    
    # Definirea transformărilor care vor fi aplicate asupra imaginilor
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Aplică transformările asupra imaginilor 
    image = data_transforms(image_array).unsqueeze(dim=0)
    
    # Creează un loader de date pentru imaginile transformate
    load = DataLoader(image)
    
    for x in load:
        # Mută datele pe dispozitiv (GPU dacă este disponibil)
        x = x.to(device)
        
        # Realizează predicții folosind modelul
        pred = model(x)
        _, preds = torch.max(pred, 1)
        
        # Verifică eticheta prezisă și returnează clasa corespunzătoare
        if preds[0] == 1:
            return "Dog"
        else:
            return "Cat"

while True:
    try:
        if __name__ == "__main__":
            image_path = 'pet_image.jpeg'
            
            # Descarcă imaginea din stocare
            storage.child('model/pet_image.jpeg').download(image_path)
            
            # Obține valoarea "compute" din baza de date care reprezintă clasificarea sau nu a imaginii animalului de companie
            compute = database.child("predictie").child("compute").get()

            if compute.val():
                # Realizează predicția imaginii
                response = PetImagePrediction("pet_image.jpeg") # imagine cu câine
                print(response)

                # Actualizează baza de date cu clasificarea
                database.child("predictie").child("tip").set(response)

                os.remove(image_path)
            else:
                response = ""
                database.child("predictie").child("tip").set(response)

    except KeyboardInterrupt:
        break
