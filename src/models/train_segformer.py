import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation_dataset import SegmentationDataset


image_dir = "data/final_segmentation_dataset/images"
mask_dir = "data/final_segmentation_dataset/masks"

dataset = SegmentationDataset(image_dir, mask_dir)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


model = smp.Segformer(
    encoder_name="mit_b0",
    encoder_weights=None,
    classes=1,
    activation=None
)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_fn = torch.nn.BCEWithLogitsLoss()

epochs = 10


for epoch in range(epochs):

    print("Epoch:", epoch+1)

    for imgs, masks in tqdm(loader):

        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


torch.save(model.state_dict(),"models/plastic_detection_model.pth")

print("Plastic detection model trained successfully")