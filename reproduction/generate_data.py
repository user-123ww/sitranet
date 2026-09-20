"""Generate data using the original upstream implementations."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def ideas(repo, arguments, preview):
    parser = argparse.ArgumentParser(prog='generate_data.py IDEAS REPO')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(arguments)
    if args.count < 1:
        parser.error('--count must be positive')
    if not (repo / 'models.py').is_file() or not (repo / 'utils.py').is_file():
        parser.error('IDEAS repository must contain models.py and utils.py')
    if preview:
        print('IDEAS: models.init_model -> utils.message_to_tensor -> Gstru_ema -> G_ema')
        print(f'checkpoint={args.checkpoint}; outdir={args.outdir}; count={args.count}; seed={args.seed}')
        return
    import os
    import random
    import numpy as np
    import torch
    from torchvision.utils import save_image
    checkpoint, outdir = (repo / args.checkpoint).resolve(), (repo / args.outdir).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from models import init_model
    from utils import message_to_tensor
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {outdir}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    settings = state['args']
    if settings.image_size != 256 or settings.N != 1:
        raise ValueError('Use a 256px, N=1 IDEAS checkpoint for 256-bit messages')
    generator = init_model('Generator', settings).to(args.device).eval()
    structure = init_model('StructureGenerator', settings).to(args.device).eval()
    generator.load_state_dict(state['trainer']['G_ema'], strict=True)
    structure.load_state_dict(state['trainer']['Gstru_ema'], strict=True)
    outdir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for offset in range(0, args.count, 16):
            count = min(16, args.count - offset)
            message = torch.randint(0, 2, (count, 256)).float()
            latent = message_to_tensor(message, sigma=1, delta=0.5).reshape(count, 1, 16, 16)
            texture = torch.rand(count, settings.texture_channel, device=args.device) * 2 - 1
            result = generator(structure(latent.to(args.device)), texture)
            for j, image in enumerate(result):
                save_image(image, outdir / f'{offset+j:06d}.png', normalize=True, value_range=(-1, 1))
    (outdir / 'generation.json').write_text(json.dumps(
        dict(method='IDEAS', checkpoint=str(args.checkpoint), count=args.count,
             seed=args.seed, message_bits=256, sigma=1, delta=0.5), indent=2) + '\n')


def main():
    entries = {'SI-SWE': 'synthesise_images_and_calculate_fid.py',
               'StyleGAN2': 'run_generator.py', 'CRoSS': 'demo.py',
               'DiffStega': 'main.py', 'mas_GRDH': 'scripts/txt2img.py'}
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Usage: python generate_data.py [--dry-run] METHOD REPO [upstream arguments]. '
        'Input/output paths are relative to the upstream working directory. Use its Python environment.')
    parser.add_argument('--dry-run', action='store_true', help='Check the entry point and print the command only')
    parser.add_argument('method', choices=['IDEAS', *entries])
    parser.add_argument('repo', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.method == 'IDEAS':
        return ideas(repo, args.arguments, args.dry_run)
    entry = repo / entries[args.method]
    if not entry.is_file():
        parser.error(f'Missing upstream entry point: {entry}')
    workdir = entry.parent if args.method == 'mas_GRDH' else repo
    entry_name = str(entry.relative_to(workdir))
    command = [sys.executable, entry_name, *args.arguments]
    display_dir = args.repo / 'scripts' if args.method == 'mas_GRDH' else args.repo
    print('cwd={}\n{}'.format(display_dir, ' '.join(
        shlex.quote(item) for item in ['python', entry_name, *args.arguments])), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=workdir, check=True)


if __name__ == '__main__':
    main()
