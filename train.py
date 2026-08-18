import os
import datetime
import random

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.optim import Adam
from utils.image_utils import integer_roll_pil, pad_and_center_crop_pil

import learn2learn as l2l

from sitranet import SITraNet, SITraNetLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = (42, 84, 168, 336, 672)
seed = int(os.environ.get('SITRANET_SEED', SEEDS[0]))
if seed not in SEEDS:
    raise ValueError(f'Unsupported seed: {seed}')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

experiments = [
    {
        'name': 'protocol_1a',
        'gamma': 1.5,
        'delta': 0.3,
        'lambda_proto': 1.5,
        'num_prototypes': 8,
        'use_multi_prototype': True,
        'proto_margin': 1.5,
        'classification_mode': 'hybrid',
        'TRAIN_DOMAINS': {
            'restaurant': 'data/train/stego/Restaurant',
            'bridge': 'data/train/stego/Bridge',
            'cat': 'data/train/stego/Cat',
        },
        'TRAIN_COVER_DIRS': 'data/train/cover',
    },
]

_IMAGE_CACHE = {}

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def _is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in _IMG_EXTS

def _list_images_recursive(root: str):
    paths = []
    if root is None:
        return paths
    if os.path.isfile(root):
        return [root] if _is_image_file(root) else []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if _is_image_file(fp):
                paths.append(fp)
    return paths

def _normalize_cover_dirs(cover_cfg, domain_keys):
    if isinstance(cover_cfg, dict):
        missing = [k for k in domain_keys if k not in cover_cfg]
        if missing:
            raise ValueError(f"Missing cover dirs for domains: {missing}")
        return cover_cfg
    return {k: cover_cfg for k in domain_keys}

class SemanticDomainDataset:
    def __init__(self, cover_paths, stego_paths, transform=None, name=""):
        self.cover_paths = cover_paths
        self.stego_paths = stego_paths
        self.transform = transform
        self.name = name

class PairedMetaDataset:
    def __init__(self,
                 cover_dirs,
                 stego_domains: dict,
                 transform=None,
                 samples_per_domain=None,
                 split=None,
                 split_ratio=0.9,
                 split_seed=42):
        self.stego_domains = stego_domains
        self.transform = transform

        domain_keys = list(stego_domains.keys())
        cover_dir_map = _normalize_cover_dirs(cover_dirs, domain_keys)

        self._domains = []
        for domain_idx, d in enumerate(domain_keys):
            s_dir = stego_domains[d]
            c_dir = cover_dir_map[d]

            stego_paths = sorted(_list_images_recursive(s_dir))
            cover_paths = sorted(_list_images_recursive(c_dir))

            random.Random(split_seed).shuffle(cover_paths)
            random.Random(split_seed + domain_idx + 1).shuffle(stego_paths)

            n = min(len(stego_paths), len(cover_paths))
            if samples_per_domain is not None:
                n = min(n, int(samples_per_domain))

            cover_paths = cover_paths[:n]
            stego_paths = stego_paths[:n]

            if split is not None:
                split_index = int(n * split_ratio)
                if split == 'train':
                    cover_paths = cover_paths[:split_index]
                    stego_paths = stego_paths[:split_index]
                elif split == 'val':
                    cover_paths = cover_paths[split_index:]
                    stego_paths = stego_paths[split_index:]
                else:
                    raise ValueError(f'Unsupported split: {split}')
                n = len(stego_paths)

            if n < 12:
                print(f"[Warning] Domain '{d}' has only {n} pairs (cover/stego). "
                      f"May cause empty tasks when sampling.")

            self._domains.append(
                SemanticDomainDataset(
                    cover_paths, stego_paths, transform=transform, name=d
                )
            )

    def __len__(self):
        return len(self._domains)

    def __getitem__(self, idx):
        return self._domains[idx]

    def __iter__(self):
        return iter(self._domains)







for i, exp_config in enumerate(experiments):
    print(f"Training {exp_config['name']}")

    experiment_name = exp_config['name']
    current_time = datetime.datetime.now().strftime('%m%d-%H%M')
    save_dir = os.path.join('outputs', f'{current_time}_{experiment_name}_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'train_log.txt')

    epochs = 200
    backbone = 'Xception'

    support_size = 12
    query_size = 12
    lr = 0.001
    inner_lr = 0.0005
    dp_rate = 0.5
    samples_per_task_per_epoch = 50
    val_samples_per_task_per_epoch = 30

    num_prototypes = exp_config.get('num_prototypes', 8)
    use_multi_prototype = exp_config.get('use_multi_prototype', True)
    proto_margin = exp_config.get('proto_margin', 1.5)
    lambda_proto = exp_config.get('lambda_proto', 1.5)
    classification_mode = exp_config.get('classification_mode', 'hybrid')

    simple_mean = [0.5, 0.5, 0.5]
    simple_std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_and_center_crop_pil(img, target=256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomRotation((90, 90), interpolation=InterpolationMode.NEAREST),
                transforms.RandomRotation((180, 180), interpolation=InterpolationMode.NEAREST),
                transforms.RandomRotation((270, 270), interpolation=InterpolationMode.NEAREST),
            ])
        ], p=0.5),
        transforms.RandomApply([
            transforms.Lambda(lambda img: integer_roll_pil(img, max_shift=3))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=simple_mean, std=simple_std),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_and_center_crop_pil(img, target=256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=simple_mean, std=simple_std),
    ])

    TRAIN_DOMAINS = exp_config['TRAIN_DOMAINS']
    COVER_DIRS = exp_config['TRAIN_COVER_DIRS']

    train_dataset = PairedMetaDataset(
        COVER_DIRS, TRAIN_DOMAINS,
        transform=train_transform,
        samples_per_domain=None,
        split='train',
        split_seed=seed,
    )

    val_dataset = PairedMetaDataset(
        COVER_DIRS, TRAIN_DOMAINS,
        transform=val_transform,
        samples_per_domain=None,
        split='val',
        split_seed=seed,
    )


    def sample_task_from_domain(domain_dataset, support_size, query_size, device):
        s_per_class = support_size // 2
        q_per_class = query_size // 2
        k_per_class = s_per_class + q_per_class

        try:
            sampled_covers_paths = random.sample(domain_dataset.cover_paths, k_per_class)
            sampled_stegos_paths = random.sample(domain_dataset.stego_paths, k_per_class)
        except ValueError:
            return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

        transform = domain_dataset.transform

        def load_with_cache(path):
            if path not in _IMAGE_CACHE:
                try:
                    with open(path, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                        img.load()
                        _IMAGE_CACHE[path] = img
                except Exception as e:
                    print(f"Error: {e}")
                    return torch.zeros((3, 256, 256))

            image = _IMAGE_CACHE[path]
            if transform:
                return transform(image)
            return image

        cover_images = [load_with_cache(p) for p in sampled_covers_paths]
        stego_images = [load_with_cache(p) for p in sampled_stegos_paths]

        support_imgs_list = cover_images[:s_per_class] + stego_images[:s_per_class]
        query_imgs_list = cover_images[s_per_class:] + stego_images[s_per_class:]

        support_labels_list = [0] * s_per_class + [1] * s_per_class
        query_labels_list = [0] * q_per_class + [1] * q_per_class

        support_imgs = torch.stack(support_imgs_list)
        support_labels = torch.tensor(support_labels_list)
        query_imgs = torch.stack(query_imgs_list)
        query_labels = torch.tensor(query_labels_list)

        s_shuffler = torch.randperm(support_imgs.size(0))
        q_shuffler = torch.randperm(query_imgs.size(0))

        return (
            support_imgs[s_shuffler].to(device),
            support_labels[s_shuffler].to(device),
            query_imgs[q_shuffler].to(device),
            query_labels[q_shuffler].to(device)
        )


    base_model = SITraNet(
        backbone=backbone,
        dropout_rate=dp_rate,
        num_prototypes=num_prototypes,
        use_multi_prototype=use_multi_prototype,
        classification_mode=classification_mode
    )
    base_model = base_model.to(device)

    maml = l2l.algorithms.MAML(base_model, lr=inner_lr, first_order=True, allow_unused=True)
    model = maml

    with open(log_path, 'w') as f:
        f.write("=" * 30 + " SITraNet + MAML Configuration " + "=" * 30 + "\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Save Directory: {save_dir}\n")
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Seed: {seed}\n")
        f.write("=" * 30 + " IMPORTANT PARAMETERS " + "=" * 30 + "\n")
        f.write(f"TRAIN_DOMAINS: {TRAIN_DOMAINS}\n")
        f.write(f"COVER_DIR: {COVER_DIRS}\n")

        f.write("\n" + "=" * 30 + " Prototype Configuration " + "=" * 30 + "\n")
        f.write(f"Number of Prototypes: {num_prototypes}\n")
        f.write(f"Use Multi-Prototype: {use_multi_prototype}\n")
        f.write(f"Proto Margin: {proto_margin}\n")
        f.write(f"Lambda Proto: {lambda_proto}\n")
        f.write(f"Classification Mode: {classification_mode}\n")
        f.write(f"\n" + "-" * 25 + " MAML Configuration " + "-" * 25 + "\n")
        f.write(f"Support Size: {support_size}\n")
        f.write(f"Query Size: {query_size}\n")
        f.write(f"Inner LR: {inner_lr}\n")
        f.write(f"Samples per Task per Epoch: {samples_per_task_per_epoch}\n")
        f.write(f"\nExp Config: {exp_config}\n")
        f.write("\n" + "=" * 30 + " Training Log " + "=" * 30 + "\n\n")

    meta_optimizer = Adam(model.parameters(), lr=lr)

    criterion_inner = SITraNetLoss(
        gamma=exp_config['gamma'],
        delta=exp_config['delta'],
        margin=proto_margin,
        lambda_proto=lambda_proto,
    )

    criterion_outer = SITraNetLoss(
        gamma=exp_config['gamma'],
        delta=exp_config['delta'],
        margin=proto_margin,
        lambda_proto=lambda_proto,
    )

    criterion = criterion_outer

    milestones = [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(meta_optimizer, milestones=milestones, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        maml.train()
        correct, total = 0, 0
        running_loss_cls = 0.0
        running_loss_rec = 0.0
        running_loss_ps = 0.0
        running_loss_proto = 0.0

        train_domain_names = list(train_dataset.stego_domains.keys())
        train_domain_stats = {name: {'correct': 0, 'total': 0} for name in train_domain_names}

        for task_idx in range(samples_per_task_per_epoch):
            meta_optimizer.zero_grad()

            if len(train_domain_names) >= 3:
                sampled_domains = random.sample(train_domain_names, 3)
            else:
                sampled_domains = random.choices(train_domain_names, k=3)

            meta_train_domains = sampled_domains[:2]
            meta_test_domain = sampled_domains[2]

            mt_imgs_list, mt_labels_list = [], []
            for d_name in meta_train_domains:
                d_idx = train_domain_names.index(d_name)
                d_dataset = train_dataset[d_idx]
                s_img, s_lbl, _, _ = sample_task_from_domain(d_dataset, support_size, query_size, device)
                if s_img.nelement() > 0:
                    mt_imgs_list.append(s_img)
                    mt_labels_list.append(s_lbl)

            if not mt_imgs_list:
                continue

            mt_imgs = torch.cat(mt_imgs_list)
            mt_labels = torch.cat(mt_labels_list)

            if epoch == 1 and task_idx == 0 and not maml.module.prototypes_initialized:
                with torch.no_grad():
                    cover_mask = (mt_labels == 0)
                    if cover_mask.sum() > 0:
                        blocks = maml.module._split_image_blocks(mt_imgs[cover_mask])
                        a1 = maml.module.encoder_a(blocks[0])
                        a2 = maml.module.encoder_a(blocks[1])
                        vec = maml.module.fusion_module(a1, a2)
                        f = maml.module.adjust_channel(vec).squeeze(-1).squeeze(-1)
                        f = F.normalize(f, p=2, dim=1)
                        maml.module.initialize_prototypes_from_data(f)

            d_idx_test = train_domain_names.index(meta_test_domain)
            d_dataset_test = train_dataset[d_idx_test]
            mts_img, mts_lbl, _, _ = sample_task_from_domain(d_dataset_test, support_size, query_size, device)

            if mts_img.nelement() == 0:
                continue

            learner = maml.clone()

            support_out = learner(mt_imgs)
            logits_f = support_out[0]

            loss_f, _ = criterion_inner(
                logits_f, mt_labels,support_out[1], support_out[2],
                support_out[3], support_out[4],
                use_cover_only=True
            )

            loss_f.backward(retain_graph=True)
            learner.adapt(loss_f, allow_nograd=True)

            del support_out, mt_imgs, mt_labels

            mts_outputs = learner(mts_img)
            logits_g = mts_outputs[0]

            loss_g, loss_dict_g = criterion_outer(
                logits_g, mts_lbl, mts_outputs[1], mts_outputs[2],
                mts_outputs[3], mts_outputs[4],
                use_cover_only=False
            )

            if torch.isnan(loss_g) or torch.isnan(loss_f):
                print(f"Warning: NaN loss detected. Skipping.")
                meta_optimizer.zero_grad()
                del learner, mts_img, mts_lbl, loss_f, loss_g
                continue

            loss_g.backward()
            meta_optimizer.step()

            batch_correct = (mts_outputs[0].argmax(dim=1) == mts_lbl).sum().item()
            batch_total = mts_lbl.size(0)


            def as_scalar(x):
                if torch.is_tensor(x):
                    return x.detach().item()
                return float(x)

            running_loss_cls += as_scalar(loss_dict_g.get('loss_cls', 0.0)) * batch_total
            running_loss_rec += as_scalar(loss_dict_g.get('loss_rec', 0.0)) * batch_total
            running_loss_ps += as_scalar(loss_dict_g.get('loss_ps', 0.0)) * batch_total
            running_loss_proto += as_scalar(loss_dict_g.get('loss_proto', 0.0)) * batch_total


            correct += batch_correct
            total += batch_total

            train_domain_stats[meta_test_domain]['correct'] += batch_correct
            train_domain_stats[meta_test_domain]['total'] += batch_total

            del learner, mts_img, mts_lbl, loss_f, loss_g

        if total > 0:
            train_acc = correct / total * 100
            train_loss_cls = running_loss_cls / total
            train_loss_rec = running_loss_rec / total
            train_loss_ps = running_loss_ps / total
            train_loss_proto = running_loss_proto / total
        else:
            train_acc = 0
            train_loss_cls = train_loss_rec = train_loss_ps = train_loss_proto = 0

        maml.eval()
        val_domain_metrics = {}
        val_domain_names = list(val_dataset.stego_domains.keys())

        total_val_loss_sum = 0.0
        total_val_samples = 0
        for domain_idx, current_val_domain in enumerate(val_dataset):
            domain_name = val_domain_names[domain_idx]
            domain_loss_sum = 0.0
            domain_correct = 0
            domain_total_samples = 0
            for _ in range(val_samples_per_task_per_epoch):

                s_imgs, s_labels, q_imgs, q_labels = sample_task_from_domain(
                    current_val_domain, support_size, query_size, device
                )

                v_imgs = torch.cat([s_imgs, q_imgs], dim=0)
                v_labels = torch.cat([s_labels, q_labels], dim=0)

                if v_imgs.nelement() == 0:
                    continue

                with torch.no_grad():
                    val_outputs = maml(v_imgs)

                    loss_v, _ = criterion(
                        val_outputs[0], v_labels, val_outputs[1], val_outputs[2],
                        val_outputs[3], val_outputs[4]
                    )

                    batch_size = v_labels.size(0)
                    domain_loss_sum += loss_v.item() * batch_size

                    preds = val_outputs[0].argmax(dim=1)
                    domain_correct += (preds == v_labels).sum().item()
                    domain_total_samples += batch_size

            if domain_total_samples > 0:
                final_domain_loss = domain_loss_sum / domain_total_samples
                final_domain_acc = (domain_correct / domain_total_samples) * 100
                total_val_loss_sum += domain_loss_sum
                total_val_samples += domain_total_samples
            else:
                final_domain_loss, final_domain_acc = 0.0, 0.0

            val_domain_metrics[domain_name] = {
                'loss': final_domain_loss,
                'acc': final_domain_acc,
            }

        if total_val_samples > 0:
            val_loss = total_val_loss_sum / total_val_samples
            val_acc = np.array([m['acc'] for m in val_domain_metrics.values()]).mean()
        else:
            val_loss, val_acc = 0.0, 0.0

        log_output = (
            f"\nEpoch {epoch:02d} | CLS:{train_loss_cls:.6f} "
            f"PROTO:{train_loss_proto:.7f} REC:{train_loss_rec:.5f} "
            f"PS:{train_loss_ps:.5f} | "
            f"\nTrain Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        print(log_output, flush=True)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_log = f"Epoch {epoch:02d} | Meta LR: {current_lr:.6f}"
        print(lr_log)


        with open(log_path, 'a') as fp:
            fp.write(f'{log_output}\n')
            fp.write(f'{lr_log}\n')

            train_domain_log = "    Train Domains: "
            for domain_name, stats in train_domain_stats.items():
                if stats['total'] > 0:
                    acc = (stats['correct'] / stats['total']) * 100
                    train_domain_log += f"[{domain_name}] {acc:.2f}% | "
            print(train_domain_log, flush=True)
            fp.write(f'{train_domain_log}\n')

            val_domain_log = "    Val Domains: "
            for domain_name, metrics in val_domain_metrics.items():
                val_domain_log += f"[{domain_name}] {metrics['acc']:.2f}% | "
            print(val_domain_log, flush=True)
            fp.write(f'{val_domain_log}\n')

        if val_acc > best_acc and epoch > int(epochs * 0.1):
            best_acc = val_acc
            torch.save(
                maml.module.state_dict(),
                os.path.join(save_dir, f"proto_maml_best_{best_acc:.2f}_ep{epoch}.pth")
            )

        elif epoch % int(epochs * 0.2) == 0 and epoch > 0:
            torch.save(
                maml.module.state_dict(),
                os.path.join(save_dir, f"proto_maml_checkpoint_ep{epoch}_{val_acc:.2f}.pth")
            )

    print(f"INFO: Experiment {exp_config['name']} finished. Results saved to {save_dir}")
    del maml, meta_optimizer
    torch.cuda.empty_cache()
