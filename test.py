import math
import os
import random

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sitranet import SITraNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'checkpoints/sitranet_ep200.pth'
MODEL_CONFIG = {
    'backbone': 'Xception',
    'dropout_rate': 0.0,
    'num_prototypes': 8,
    'use_multi_prototype': True,
    'classification_mode': 'hybrid',
}

STEGO_TEST_DOMAINS = {
    'bedroom': 'test_data/Bedroom/stego',
    'church': 'test_data/Church/stego',
    'celeba': 'test_data/CelebA/stego',
}

STEGO_COVER_DIRS = {
    'bedroom': 'test_data/Bedroom/cover',
    'church': 'test_data/Church/cover',
    'celeba': 'test_data/CelebA/cover',
}

TEST_SAMPLES = 500
BATCH_SIZE = 64
SEED = 42


def pad_and_center_crop_pil(img, target=256):
    w, h = img.size
    if w == target and h == target:
        return img
    if w < target or h < target:
        padding = (
            (target - w) // 2,
            (target - h) // 2,
            target - w - (target - w) // 2,
            target - h - (target - h) // 2,
        )
        img = ImageOps.expand(img, padding, fill=0)
    w, h = img.size
    return img.crop(((w - target) / 2, (h - target) / 2,
                     (w + target) / 2, (h + target) / 2))


class TestDataset(Dataset):
    def __init__(self, cover_dir, stego_dir, transform, length, seed):
        self.transform = transform
        random.seed(seed)

        self.cover_paths = self._list_images(cover_dir)
        self.stego_paths = self._list_images(stego_dir)
        random.shuffle(self.cover_paths)
        random.shuffle(self.stego_paths)

        length = min(length, len(self.stego_paths))
        if not self.cover_paths:
            raise ValueError(f'Cover directory {cover_dir} is empty.')
        if len(self.cover_paths) < length:
            repeat_factor = math.ceil(length / len(self.cover_paths))
            self.cover_paths = (self.cover_paths * repeat_factor)[:length]
        else:
            self.cover_paths = self.cover_paths[:length]
        self.stego_paths = self.stego_paths[:length]

        length = min(len(self.cover_paths), len(self.stego_paths))
        self.paths = self.cover_paths[:length] + self.stego_paths[:length]
        self.labels = [0] * length + [1] * length

    @staticmethod
    def _list_images(directory):
        paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp')):
                    paths.append(os.path.join(root, filename))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image), torch.tensor(self.labels[index], dtype=torch.long)


def load_model():
    model = SITraNet(**MODEL_CONFIG)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = {
            key[7:] if key.startswith('module.') else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_test():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_and_center_crop_pil(img, target=256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    model = load_model()
    domain_accuracies = []

    for domain, stego_dir in STEGO_TEST_DOMAINS.items():
        dataset = TestDataset(
            cover_dir=STEGO_COVER_DIRS[domain],
            stego_dir=stego_dir,
            transform=transform,
            length=TEST_SAMPLES,
            seed=SEED,
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                outputs, *_ = model(images.to(device))
                predictions = outputs.argmax(dim=1).cpu()
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        domain_accuracies.append(100.0 * correct / total)

    return sum(domain_accuracies) / len(domain_accuracies)


if __name__ == '__main__':
    print(f'Average ACC: {run_test():.2f}%')
