import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

from .dataset import Pix2PixDataset
from .model import GeneratorUNet, Discriminator


def train_pix2pix(tree_ids=None, epochs: int = 100, batch_size: int = 4, lr: float = 0.0002, device='cuda'):
    """
    Обучение Pix2Pix на данных всех указанных деревьев.
    Если tree_ids=None, используются все деревья с REAL уровнями.
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Используем устройство: {device}")

    dataset = Pix2PixDataset(tree_ids, min_height_diff=5, max_height_diff=50, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    g_optim = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # Папка для сохранения модели – теперь общая для всех деревьев
    checkpoint_dir = Path("data/models/pix2pix/all_trees")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        for i, (input_img, target_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            # --- Обучение генератора ---
            g_optim.zero_grad()
            fake = generator(input_img)
            pred_fake = discriminator(input_img, fake)
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_l1 = criterion_l1(fake, target_img) * 100
            loss_g = loss_gan + loss_l1
            loss_g.backward()
            g_optim.step()

            # --- Обучение дискриминатора ---
            d_optim.zero_grad()
            pred_real = discriminator(input_img, target_img)
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            pred_fake = discriminator(input_img, fake.detach())
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_d = (loss_real + loss_fake) * 0.5
            loss_d.backward()
            d_optim.step()

        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
            }, checkpoint_dir / f"epoch_{epoch+1}.pth")
            print(f"Сохранён чекпоинт epoch_{epoch+1}.pth")

    torch.save(generator.state_dict(), checkpoint_dir / "generator_final.pth")
    print(f"Обучение завершено! Модель сохранена в {checkpoint_dir}")