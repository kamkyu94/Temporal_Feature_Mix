import os
import cv2
import math
import torch
import kornia
import ctypes
import os.path
import warnings
import accimage
import numpy as np
import torchvision
import collections
import skimage as sk
import torch.nn as nn
from PIL import Image
from io import BytesIO
from numba import njit
from shutil import copyfile
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image as PILImage
from typing import Tuple, Optional
from skimage.filters import gaussian
import torchvision.transforms as trn
from scipy.ndimage import zoom as scizoom
from wand.image import Image as WandImage
from wand.api import library as wandlibrary


warnings.simplefilter("ignore", UserWarning)
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# Tell Python about the C method (wand, radius, sigma, angle)
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, severity, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.method = method
        self.severity = severity
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.last_seq = ''
        self.cnt_dict = {"MOT17-04-FRCNN": 0, "MOT17-05-FRCNN": 0, "MOT17-09-FRCNN": 0, "MOT20-01": 0, "MOT20-03": 0}
        self.max_cnt = 30000
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        self.cnt_dict[self.idx_to_class[target]] += 1
        if self.cnt_dict[self.idx_to_class[target]] < self.max_cnt:
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
                if 'image_stack' in self.method.__name__:
                    img = self.method(img, self.severity, self.idx_to_class[target])
                else:
                    img = self.method(img, self.severity)
            if self.target_transform is not None:
                target = self.target_transform(target)

            save_path = '../dataset/MOT-C/' + self.method.__name__ + '_' + str(self.severity) \
                        + '/' + self.idx_to_class[target] + '/img1'
            dst_path = '../dataset/MOT-C/' + self.method.__name__ + '_' + str(self.severity) + \
                       '/' + self.idx_to_class[target]+'/seqinfo.ini'
            src_path = '../dataset/MOT-C/test/' + self.idx_to_class[target] + '/seqinfo.ini'

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if not os.path.exists(dst_path):
                copyfile(src_path, dst_path)
            save_path += path[path.rindex('/'):]
            print(path[path.rindex('/'):], flush=True)

            Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)

        return 0

    def __len__(self):
        return len(self.imgs)


def my_gaussian(window_size, sigma):
    x = torch.arange(window_size).float().to(device=sigma.device) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. Got {}".format(kernel_size))
    window_1d: torch.Tensor = my_gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tuple[float, float],
                          force_even: bool = False) -> torch.Tensor:
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}".format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))

    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())

    return kernel_2d


def apply_gaussian(tensor, sigma):
    kernel_size = int(2*(4.0*sigma+0.5))
    k = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
    t2 = kornia.filters.filter2d(tensor[:, None], kernel=k[None], border_type='constant')
    return t2


def elastic_transform_2d(tensor: torch.Tensor, sigma: Tuple[float, float] = (4., 4.),
                         alpha: Tuple[float, float] = (32., 32.), random_seed: Optional = None) -> torch.Tensor:
    generator = torch.Generator(device='cpu')

    if random_seed is not None:
        generator.manual_seed(random_seed)

    n, c, h, w = tensor.shape

    # Convolve over a random displacement matrix and scale them with 'alpha'
    d_rand = torch.rand(n, 2, h, w, generator=generator).to(
        device=tensor.device) * 2 - 1

    tensor_y = d_rand[:, 0]
    tensor_x = d_rand[:, 1]

    dy = apply_gaussian(tensor_y, sigma[0]) * alpha[0]
    dx = apply_gaussian(tensor_x, sigma[1]) * alpha[1]

    # stack and normalize displacement
    d_yx = torch.cat([dy, dx], dim=1).permute(0, 2, 3, 1)

    # Warp image based on displacement matrix
    grid = kornia.utils.create_meshgrid(h, w).to(device=tensor.device)
    warped = F.grid_sample(
        tensor, (grid + d_yx).clamp(-1, 1), align_corners=True)

    return warped


class ElasticTransform(nn.Module):
    def __init__(self, alpha=1, sigma=12, random_seed=42):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(2), requires_grad=True) * math.log(sigma)
        self.log_alpha = nn.Parameter(torch.ones(2), requires_grad=True) * math.log(alpha)
        self.random_seed = random_seed

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        alpha = torch.exp(self.log_alpha)
        img_transformed = elastic_transform_2d(x, sigma=sigma, alpha=alpha,
                                               random_seed=self.random_seed)
        return img_transformed


def elastic_transform(image, severity=1):
    rs = np.random.randint(0,10000)
    image = np.array(image, dtype=np.float32) / 255.
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    et = ElasticTransform(alpha=0.04 * severity,  sigma=12, random_seed=rs)
    image_transformed = et.forward(image)[0]
    img = np.array(image_transformed.cpu().detach().numpy()).transpose(1, 2, 0) * 255.
    return img


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 5][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


class ImageStacker(object):
    def __init__(self) -> None:
        self.last_h = 0
        self.last_w = 0
        self.last_seq = ''
        self.save_images = []
        self.fps_dict = {"MOT17-04-FRCNN": 30, "MOT17-05-FRCNN": 14,
                         "MOT17-09-FRCNN": 30, "MOT20-01": 25, "MOT20-03": 25}
        pass

    def image_stack(self,x, severity=1,seq=''):
        s = [14, 12, 10, 8, 6][severity - 1]
        ml = self.fps_dict[seq]
        II = np.array(x, dtype=np.float32) / 255.
        h, w, c = II.shape

        if len(self.save_images) >= ml:
            self.save_images.pop(0)
        self.save_images.append(II)
        if self.last_seq != seq:
            self.save_images = [II]
        if h != self.last_h or w != self.last_w:
            self.save_images = [II]
        
        self.last_h = h
        self.last_w = w
        out = np.zeros_like(x, dtype=np.float32)
        weight = s
        w = np.exp((np.arange(0, ml) / ml - 1) * weight)
        ww = np.sum(w[ml - len(self.save_images):ml])
        for i in range(len(self.save_images)):
            out += w[ml - i - 1] / ww * self.save_images[len(self.save_images) - 1 - i]
        self.last_seq = seq

        return np.clip(out, 0, 1) * 255


# Numba nopython compilation to shuffle_pixles
@njit()
def _shuffle_pixels_njit_glass_blur(d0, d1, x, c):
    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
    x = _shuffle_pixels_njit_glass_blur(np.array(x).shape[0], np.array(x).shape[1], x, c)
    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (18, 14)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        # BGR to RGB
        return np.clip(x[..., [2, 1, 0]], 0, 255)
    else:
        # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    cw = int(np.ceil(w / zoom_factor))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top:trim_top + h, trim_left:trim_left + w]


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.01, 0.001),
         np.arange(1, 1.02, 0.002),
         np.arange(1, 1.03, 0.003),
         np.arange(1, 1.04, 0.004),
         np.arange(1, 1.05, 0.005)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x, severity=1):
    c = [(0.5, 2.5), (0.75, 2), (1., 2), (1.25, 1.7), (1.5, 1.5)][severity - 1]
    x = np.array(x) / 255.
    max_val = x.max()
    hh = x.shape[0]
    ww = x.shape[1]
    length = 2**(math.ceil(math.log(max(hh,ww))/math.log(2)))
    x += c[0] * plasma_fractal(mapsize=length, wibbledecay=c[1])[:hh, :ww][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(0.95, 0.1), (0.9, 0.2), (0.85, 0.3), (0.8, 0.4), (0.75, 0.5)][severity - 1]
    idx = np.random.randint(5)
    xx = np.array(x) / 255.
    hh = xx.shape[1]
    ww = xx.shape[0]

    filename = ['./frost_images/frost1.jpg', './frost_images/frost2.jpg', './frost_images/frost3.jpg',
                './frost_images/frost4.jpg', './frost_images/frost5.jpg', './frost_images/frost6.jpg',
                './frost_images/frost7.jpg'][idx]
    frost = cv2.imread(filename)

    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - ww), np.random.randint(0, frost.shape[1] - hh)
    frost = frost[x_start:x_start + ww, y_start:y_start + hh][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.7, 10, 4, 0.9), (0.1, 0.3, 3, 0.6, 10, 4, 0.8), (0.2, 0.3, 2, 0.5, 12, 4, 0.8),
         (0.55, 0.3, 4, 0.9, 12, 6, 0.7), (0.55, 0.3, 4.5, 0.85, 12, 8,  0.7)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.

    hh = x.shape[0]
    ww = x.shape[1]
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(hh, ww, 1) * 1.5 + 0.5)

    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0), (0.65, 0.3, 3, 0.68, 0.6, 0), (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1), (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)

        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]), 238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255

    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    image = np.array(x, dtype=np.float32) / 255.
    ww = image.shape[0]
    hh = image.shape[1]
    x = x.resize((int(hh * c), int(ww * c)), PILImage.BOX)
    x = x.resize((hh, ww), PILImage.BOX)

    return x


def save_distorted(method=gaussian_noise):
    for severity in [1, 2, 3, 4, 5]:
        print(method.__name__, severity)
        distorted_dataset = DistortImageFolder( root="../dataset/MOT-C/test/", method=method, severity=severity,
                                                transform=trn.Compose([]))
        distorted_dataset_loader = torch.utils.data.DataLoader(distorted_dataset, batch_size=100,
                                                               shuffle=False, num_workers=1)

        for _ in distorted_dataset_loader:
            continue


print('\nUsing MOT data')

IS = ImageStacker()
d = collections.OrderedDict()

d['JPEG'] = jpeg_compression
d['Motion Blur'] = motion_blur
d['Elastic'] = elastic_transform
d['Impulse Noise'] = impulse_noise
d['Gaussian Noise'] = gaussian_noise
d['Shot Noise'] = shot_noise
d['Speckle Noise'] = speckle_noise
d['Gaussian Blur'] = gaussian_blur
d['Spatter'] = spatter
d['Image Stack'] = IS.image_stack
d['Fog'] = fog
d['Snow'] = snow
d['Frost'] = frost
d['Pixelate'] = pixelate
d['Zoom Blur'] = zoom_blur
d['Glass Blur'] = glass_blur

for method_name in d.keys():
    save_distorted(d[method_name])
