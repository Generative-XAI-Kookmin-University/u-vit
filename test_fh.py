import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from libs.flaw_highlighter import FlawHighlighter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_horizontal_flip=False,
            convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if convert_image_to else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def test_flaw_highlighter(FH, real_test_path, fake_test_path, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FH = FH.to(device)
    FH.eval()

    # Load test datasets
    real_dataset = Dataset(real_test_path, 64)
    fake_dataset = Dataset(fake_test_path, 64)
    
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    # Evaluate real images
    print("Evaluating real images")
    with torch.no_grad():
        for real_images in tqdm(real_loader):
            real_images = real_images.to(device)
            logits = FH(real_images)
            preds = logits[:, 1]
            preds = preds.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(np.ones(len(real_images)))

    # Evaluate fake images
    print("Evaluating generated images")
    with torch.no_grad():
        for fake_images in tqdm(fake_loader):
            fake_images = fake_images.to(device)
            logits = FH(fake_images)
            preds = logits[:, 1]
            preds = preds.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(np.zeros(len(fake_images)))

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_rounded = np.round(all_preds)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, preds_rounded)
    f1 = f1_score(all_labels, preds_rounded)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

if __name__ == '__main__':
    params = {
        'nc': 3,
        'ndf': 32,
    }

    FH = FlawHighlighter(params)
    ckpt_path = 'fh_ckpt/FH_best_24.pth'
    ckpt = torch.load(ckpt_path)
    FH.load_state_dict(ckpt['model_state_dict'])

    # Test paths
    real_test_path = './data/celeba-FH/real_test'
    fake_test_path = './data/celeba-FH/fake_test'

    # Run evaluation
    metrics = test_flaw_highlighter(FH, real_test_path, fake_test_path)