import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import torch.backends.cudnn as cudnn
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2


def robust_radius_binary(net, x, per_gen, p0, p1, delta, eps, left, right):
    delta = delta / (math.ceil(math.log2((right - left) / eps)))
    while right - left >= eps:
        now = (left + right) / 2
        if sequential(net, x, per_gen, p0, p1, delta, delta, now) == 'Yes':
            # p<p1, increase the perturbation
            left = now
        else:
            # p>p0, decrease the perturbation
            right = now
    return (left + right) / 2


def sequential(net, x, per_gen, p0, p1, alpha, beta, per_size):
    dm = 0
    m = 0
    k1 = math.log(p1 / p0)
    k2 = math.log((1 - p1) / (1 - p0))
    Aln = math.log(1 / alpha)
    Bln = math.log(beta)
    batch_size = 100
    while True:
        result = evaluate(net, x, per_gen, per_size, batch_size)
        for i in range(batch_size):
            m += 1
            if not result[i]:  # adversarial
                dm += 1
            fm = dm * k1 + (m - dm) * k2
            if fm > Aln:  # p<p1
                return 'Yes'
            elif fm < Bln:  # p>p0
                return 'No'


def evaluate(_net, x, per_gen, per_size, batch_size):
    per_x = per_gen(x, per_size, batch_size)
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    x = transform(x)
    per_x = transform(per_x)
    _net.eval()
    with torch.no_grad():
        x, per_x = x.to(device), per_x.to(device)
        outputs_x = _net(x)
        _, predicted_x = outputs_x.max(1)
        outputs = _net(per_x)
        _, predicted = outputs.max(1)
        result = (predicted == predicted_x)
    return result


def gauss_noise(x, per_size, num):
    x = x.repeat(num, 1, 1, 1).cpu()
    return np.clip(x + np.random.normal(size=x.shape, scale=per_size).astype('float32'), 0, 1)


def contrast(x, per_size, num):
    c = np.random.uniform(1-per_size, 1, (num, 1, 1, 1))
    x = x.repeat(num, 1, 1, 1).cpu()
    means = np.mean(x.numpy(), axis=(2, 3), keepdims=True)
    return np.clip((x[0] - means[0]) * c + means, 0, 1).float()


def plasma_fractal(mapsize=32, wibbledecay=3):
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


def fog(x, per_size, num):
    c = (1+per_size*0.5, 2-per_size*0.25)
    x = x.repeat(num, 1, 1, 1).cpu()
    max_val = x.max()
    for i in range(num):
        x[i] += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][np.newaxis, ...]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1).float()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_num = 8
net = [None] * net_num
# original networks
net[0] = VGG('VGG16')
net[1] = DenseNet121()
net[2] = ResNet101()
net[3] = MobileNetV2()
# adversarial trained networks
net[4] = VGG('VGG16')
net[5] = DenseNet121()
net[6] = ResNet101()
net[7] = MobileNetV2()
for i in range(net_num):
    net[i] = net[i].to(device)
    if device == 'cuda':
        net[i] = torch.nn.DataParallel(net[i])
        cudnn.benchmark = True
weight = torch.load('net_weight/' + 'vgg16' + '_ckpt.pth')
net[0].load_state_dict(weight['net'])
net[0].eval()
weight = torch.load('net_weight/' + 'densenet121' + '_ckpt.pth')
net[1].load_state_dict(weight['net'])
net[1].eval()
weight = torch.load('net_weight/' + 'resnet101' + '_ckpt.pth')
net[2].load_state_dict(weight['net'])
net[2].eval()
weight = torch.load('net_weight/' + 'mobilenetv2' + '_ckpt.pth')
net[3].load_state_dict(weight['net'])
net[3].eval()

weight = torch.load('net_weight/robust_GN_' + 'vgg16' + '_ckpt.pth')
net[4].load_state_dict(weight)
net[4].eval()
weight = torch.load('net_weight/robust_GN_' + 'densenet121' + '_ckpt.pth')
net[5].load_state_dict(weight)
net[5].eval()
weight = torch.load('net_weight/robust_GN_' + 'resnet101' + '_ckpt.pth')
net[6].load_state_dict(weight)
net[6].eval()
weight = torch.load('net_weight/robust_GN_' + 'mobilenetv2' + '_ckpt.pth')
net[7].load_state_dict(weight)
net[7].eval()

# weight = torch.load('net_weight/contrast_' + 'vgg16' + '_ckpt.pth')
# net[4].load_state_dict(weight)
# net[4].eval()
# weight = torch.load('net_weight/contrast_' + 'densenet121' + '_ckpt.pth')
# net[5].load_state_dict(weight)
# net[5].eval()
# weight = torch.load('net_weight/contrast_' + 'resnet101' + '_ckpt.pth')
# net[6].load_state_dict(weight)
# net[6].eval()
# weight = torch.load('net_weight/contrast_' + 'mobilenetv2' + '_ckpt.pth')
# net[7].load_state_dict(weight)
# net[7].eval()

# weight = torch.load('net_weight/fg_' + 'vgg16' + '_ckpt.pth')
# net[4].load_state_dict(weight)
# net[4].eval()
# weight = torch.load('net_weight/fg_' + 'densenet121' + '_ckpt.pth')
# net[5].load_state_dict(weight)
# net[5].eval()
# weight = torch.load('net_weight/fg_' + 'resnet101' + '_ckpt.pth')
# net[6].load_state_dict(weight)
# net[6].eval()
# weight = torch.load('net_weight/fg_' + 'mobilenetv2' + '_ckpt.pth')
# net[7].load_state_dict(weight)
# net[7].eval()

transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
dataiter = iter(testloader)

result_radius = [[0 for i in range(10)] for j in range(net_num)]
for ii in range(10):
    while True:
        images, labels = dataiter.next()
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            images_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(images)
            all_correct = True
            for i in range(net_num):
                if net[i](images_norm).max(1)[1][0] != labels:
                    all_correct = False
                    break
            if all_correct:
                break

    for i in range(net_num):
        result_radius[i][ii] = robust_radius_binary(net[i], images, gauss_noise, 0.015, 0.005, 0.01, 0.001, 0, 0.2)
        # result_radius[i][ii] = robust_radius_binary(net[i], images, contrast, 0.015, 0.005, 0.01, 0.001, 0, 1)
        # result_radius[i][ii] = robust_radius_binary(net[i], images, fog, 0.015, 0.005, 0.01, 0.001, 0, 1)

print(result_radius)
