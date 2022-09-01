import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import torch.backends.cudnn as cudnn
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
import time


# Our method
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


# Code for PROVERO comes from https://github.com/teobaluta/provero
# PROVERO-interval generate
def divide_et_impera(theta, eta, theta1, theta2, left):
    if theta == 0 and left:
        return theta, theta + eta
    if theta == 1:
        print('Always true xD!')
        exit(1)

    if theta1 == 0 and theta2 == 0 and left:
        return 0, theta

    if theta1 == 0 and theta2 == 0 and not left:
        return theta + eta, 1

    alpha = theta2 - theta1
    if left:
        return theta2 - max(eta, alpha / 2), theta2

    return theta1, theta1 + max(eta, alpha / 2)


# PROVERO-main function
def adaptive_assert(net, x, per_gen, theta, eta, delta, per_size,
                    interval_strategy=divide_et_impera):
    theta1_left = 0
    theta2_left = 0
    theta1_right = 0
    theta2_right = 0

    # adjust the delta
    n = 3 + max(0, math.log2(theta / eta)) + max(0, math.log2((1 - theta - eta) / eta))
    delta = delta / n

    while True:
        theta1_left, theta2_left = interval_strategy(theta, eta, theta1_left, theta2_left, True)
        assert (theta1_left < theta2_left)
        assert (theta >= theta2_left)
        if theta2_left - theta1_left > eta + 1e-8:
            tester_ans = tester(net, x, per_gen, theta1_left, theta2_left, delta, per_size)
            if tester_ans == 'Yes':
                return 'Yes'

        theta1_right, theta2_right = interval_strategy(theta, eta, theta1_right,
                                                       theta2_right, False)
        assert (theta1_right < theta2_right)
        assert (theta + eta <= theta1_right)
        if theta2_right - theta1_right - eta <= 1e-8:
            if theta2_left - theta1_left <= eta + 1e-8:
                tester_ans = tester(net, x, per_gen, theta, theta + eta, delta, per_size)
                return tester_ans
        else:
            tester_ans = tester(net, x, per_gen, theta1_right, theta2_right, delta, per_size)
            if tester_ans == 'No':
                return 'No'


# PROVERO-TESTER
def tester(net, x, per_gen, theta1, theta2, delta, per_size):
    global doc

    assert (theta2 > theta1)
    if theta2 > 1:
        theta2 = 1
    if theta1 < 0:
        theta1 = 0

    # compute number of samples
    N = 1 / ((theta2 - theta1) ** 2) * math.log(1 / delta) * (math.sqrt(3 * theta1) + math.sqrt(2 * theta2)) ** 2
    N = math.ceil(N)

    s = 0
    c = 0
    batch_size = 100
    while True:
        if batch_size + c < N:
            result = evaluate(net, x, per_gen, per_size, batch_size)
            s += result.eq(False).sum().item()
            c += batch_size
        else:
            result = evaluate(net, x, per_gen, per_size, N - c)
            s += result.eq(False).sum().item()
            break
    mean = s / N
    if mean <= theta1 + (theta2 - theta1) * math.sqrt(3 * theta1) / (
            math.sqrt(3 * theta1) + math.sqrt(2 * theta2)):
        return 'Yes'
    else:
        return 'No'


# One-side method (arXiv:2010.07532)
def one_side(net, x, per_gen, p0, delta, per_size):
    N = 2*(math.log(1/delta)+1)/p0
    N = math.ceil(N)
    c = 0
    batch_size = 100
    while True:
        if batch_size + c < N:
            result = evaluate(net, x, per_gen, per_size, batch_size)
            if result.eq(False).sum().item() > 0:
                return 'No'
            c += batch_size
        else:
            result = evaluate(net, x, per_gen, per_size, N - c)
            if result.eq(False).sum().item() > 0:
                return 'No'
            break
    return 'Yes'


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
    c = np.random.uniform(1-per_size, 1+per_size, (num, 1, 1, 1))
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
    x = x.repeat(num, 1, 1, 1).cpu()
    max_val = x.max()
    for i in range(num):
        x[i] += per_size[0] * plasma_fractal(wibbledecay=per_size[1])[:32, :32][np.newaxis, ...]
    return np.clip(x * max_val / (max_val + per_size[0]), 0, 1).float()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_num = 10
net = VGG('VGG16')  # fog
# net = DenseNet121()  # contrast
# net = ResNet101()  # gauss
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
weight = torch.load('net_weight/' + 'vgg16' + '_ckpt.pth')  # fog
# weight = torch.load('net_weight/' + 'densenet121' + '_ckpt.pth')  # contrast
# weight = torch.load('net_weight/' + 'resnet101' + '_ckpt.pth')  # gauss
net.load_state_dict(weight['net'])
net.eval()

transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
dataiter = iter(testloader)

while True:
    images, labels = dataiter.next()
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        images_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(images)
        if net(images_norm).max(1)[1][0] == labels:
            print(labels)
            break

p0_list = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18,
           0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
           0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
delta = 0.01  # confidence
eta = 0.01  # p0-p1
per_size = (1.5, 1.75)  # fog
# per_size = 1.0  # contrast
# per_size = 0.08  # gauss
start = time.time()
for i in range(len(p0_list)):
    p0 = p0_list[i]
    print(sequential(net, images, fog, p0, p0 - eta, delta, delta, per_size), p0)
    # print(sequential(net, images, contrast, p0, p0 - eta, delta, delta, per_size), p0)
    # print(sequential(net, images, gauss_noise, p0, p0 - eta, delta, delta, per_size), p0)
end = time.time()
print(end-start)

start = time.time()
for i in range(len(p0_list)):
    p0 = p0_list[i]
    print(adaptive_assert(net, images, fog, p0 - eta, eta, delta, per_size), p0)
    # print(adaptive_assert(net, images, contrast, p0 - eta, eta, delta, per_size), p0)
    # print(adaptive_assert(net, images, gauss_noise, p0 - eta, eta, delta, per_size), p0)
end = time.time()
print(end-start)

start = time.time()
for i in range(len(p0_list)):
    p0 = p0_list[i]
    print(one_side(net, images, fog, p0, delta, per_size), p0)
    # print(one_side(net, images, contrast, p0, delta, per_size), p0)
    # print(one_side(net, images, gauss_noise, p0, delta, per_size), p0)
end = time.time()
print(end-start)
