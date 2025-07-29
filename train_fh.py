import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from libs.flaw_highlighter import FlawHighlighter
from tqdm import tqdm
import wandb
import os

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


class FHTrainer(object):
    def __init__(self, FH, real_img, gen_img, image_size, FH_ckpt=None, batch_size=64, lr=2e-5, adam_betas=(0.5, 0.999), num_epoch=30):
        super().__init__()

        self.FH = FH
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epoch = num_epoch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FH = self.FH.to(self.device)
        print('device:', self.device)

        if FH_ckpt:
            ckpt = torch.load(FH_ckpt)
            ckpt_state_dict = ckpt["model_state_dict"]
            model_state_dict = self.FH.state_dict()

            filtered_state_dict = {}
            for k, v in ckpt_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Shape mismatch for {k}: checkpoint {v.shape}, model {model_state_dict[k].shape}")
                else:
                    print(f"Unexpected key in checkpoint: {k}")

            model_state_dict.update(filtered_state_dict)
            self.FH.load_state_dict(model_state_dict, strict=False)
            print('checkpoint accuracy:', ckpt['accuracy'])

        self.criterion = nn.NLLLoss()

        self.real_dataset = Dataset(real_img, self.image_size)
        self.gen_dataset = Dataset(gen_img, self.image_size)

        real_data_size = len(self.real_dataset)
        gen_data_size = len(self.gen_dataset)

        real_test_size = int(real_data_size * 0.1)
        gen_test_size = int(gen_data_size * 0.1)

        self.train_real, self.test_real = random_split(self.real_dataset, [real_data_size - real_test_size, real_test_size])
        self.train_gen, self.test_gen = random_split(self.gen_dataset, [gen_data_size - gen_test_size, gen_test_size])

        self.train_real_dl = DataLoader(self.train_real, batch_size=self.batch_size, shuffle=True)
        self.test_real_dl = DataLoader(self.test_real, batch_size=self.batch_size, shuffle=False)
        self.train_gen_dl = DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True)
        self.test_gen_dl = DataLoader(self.test_gen, batch_size=self.batch_size, shuffle=False)

        self.opt = AdamW(self.FH.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler = OneCycleLR(self.opt, max_lr=self.lr, epochs=self.num_epoch, steps_per_epoch=len(self.train_real_dl))

        self.best_acc = 0
        
    def train(self):
        wandb.init(project="uvit-fh")

        for epoch in range(1, self.num_epoch + 1):
            self.FH.train()
            epoch_loss = []

            pbar = tqdm(zip(self.train_real_dl, self.train_gen_dl), total=len(self.train_real_dl), desc=f"Epoch {epoch}/{self.num_epoch}")

            for real_images, gen_images in pbar:
                real_images, gen_images = real_images.to(self.device), gen_images.to(self.device)

                real_labels = torch.ones(len(real_images), dtype=torch.long, device=self.device)
                fake_labels = torch.zeros(len(gen_images), dtype=torch.long, device=self.device)

                real_outputs = self.FH(real_images)
                fake_outputs = self.FH(gen_images)

                log_real_outputs = torch.log(real_outputs + 1e-9)
                log_fake_outputs = torch.log(fake_outputs + 1e-9)

                real_loss = self.criterion(log_real_outputs, real_labels)
                fake_loss = self.criterion(log_fake_outputs, fake_labels)

                total_loss = (real_loss + fake_loss) / 2

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                self.scheduler.step()

                epoch_loss.append(total_loss.item())
                pbar.set_postfix({'loss': total_loss.item()})
                wandb.log({"iteration loss": total_loss})

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            wandb.log({"epoch loss": avg_epoch_loss})

            print(f"Epoch [{epoch}/{self.num_epoch}], Loss: {avg_epoch_loss}")

            if not os.path.exists('./fh_ckpt/'):
                os.makedirs('./fh_ckpt/')
            
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': self.FH.state_dict(), 
            #     'optimizer_state_dict': self.opt.state_dict(),  
            #     'scheduler_state_dict': self.scheduler.state_dict(),
            #     'loss': avg_epoch_loss 
            # }, f'./fh_ckpt/FH_{epoch}.pth')

            self.FH.eval()
            all_real_preds = []
            all_fake_preds = []
            all_real_labels = []
            all_fake_labels = []

            with torch.no_grad():
                for real_images, generated_images in zip(self.test_real_dl, self.test_gen_dl):
                    real_images, generated_images = real_images.to(self.device), generated_images.to(self.device)

                    real_outputs = FH(real_images)
                    fake_outputs = FH(generated_images)
                
                    real_preds = real_outputs.argmax(dim=1).cpu().detach().numpy()
                    fake_preds = fake_outputs.argmax(dim=1).cpu().detach().numpy()
                
                    real_labels = np.ones(len(real_images))
                    fake_labels = np.zeros(len(generated_images))
                
                    all_real_preds.extend(real_preds)
                    all_fake_preds.extend(fake_preds)
                    all_real_labels.extend(real_labels)
                    all_fake_labels.extend(fake_labels)
                    
            real_preds = np.array(all_real_preds)
            fake_preds = np.array(all_fake_preds)
            real_labels = np.array(all_real_labels)
            fake_labels = np.array(all_fake_labels)

            real_preds_rounded = np.round(real_preds)
            fake_preds_rounded = np.round(fake_preds)

            real_acc = accuracy_score(real_labels, real_preds_rounded)
            fake_acc = accuracy_score(fake_labels, fake_preds_rounded)
            avg_acc = (real_acc + fake_acc) / 2

            avg_f1 = f1_score(np.concatenate([real_labels, fake_labels]), np.concatenate([real_preds_rounded, fake_preds_rounded]))

            avg_roc_auc = roc_auc_score(np.concatenate([real_labels, fake_labels]), np.concatenate([real_preds, fake_preds]))

            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.FH.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    'roc_auc': avg_roc_auc,
                    'accuracy': avg_acc,
                    'f1_score': avg_f1
                }, f'./fh_ckpt/FH_best_{epoch}.pth')
            

if __name__ == '__main__':

    params = {
        'nc' : 3,
        'ndf' : 32,
        }

    FH = FlawHighlighter(params)
    FH_ckpt = './FH_best_7_share.pth'

    FH_trainer = FHTrainer(FH=FH,
                        real_img='./data/celeba-FH/real_train/',
                        gen_img='./data/celeba-FH/fake_train/',
                        image_size=64,
                        FH_ckpt = FH_ckpt)

    FH_trainer.train()