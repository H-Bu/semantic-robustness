import math
import time
import random

count = 0


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
def adaptive_assert(theta, eta, delta, obj_to_sample, timeout, p,
                    interval_strategy=divide_et_impera):
    if obj_to_sample is None:
        return

    theta1_left = 0
    theta2_left = 0
    theta1_right = 0
    theta2_right = 0

    # adjust the delta
    n = 3 + max(0, math.log2(theta / eta)) + max(0, math.log2((1 - theta - eta) / eta))
    delta = delta / n

    start_time = time.time()
    while True:
        if time.time() - start_time >= timeout:
            print('timeout!')
            return 'timeout'

        theta1_left, theta2_left = interval_strategy(theta, eta, theta1_left, theta2_left, True)
        assert (theta1_left < theta2_left)
        assert (theta >= theta2_left)
        if theta2_left - theta1_left > eta + 1e-8:
            tester_ans = tester(theta1_left, theta2_left, delta, obj_to_sample,
                                timeout - (time.time() - start_time), p)
            if tester_ans is 'timeout':
                return 'timeout'
            if tester_ans is 'Yes':
                return 'Yes'

        theta1_right, theta2_right = interval_strategy(theta, eta, theta1_right,
                                                       theta2_right, False)
        assert (theta1_right < theta2_right)
        assert (theta + eta <= theta1_right)
        if theta2_right - theta1_right - eta <= 1e-8:
            if theta2_left - theta1_left <= eta + 1e-8:
                tester_ans = tester(theta, theta + eta, delta, obj_to_sample,
                                    timeout - (time.time() - start_time), p)
                return tester_ans
        else:
            tester_ans = tester(theta1_right, theta2_right, delta, obj_to_sample,
                                timeout - (time.time() - start_time), p)
            if tester_ans is 'No':
                return 'No'
            if tester_ans is 'timeout':
                return 'timeout'


# PROVERO-TESTER
def tester(theta1, theta2, delta, obj_to_sample, timeout, p):
    assert (theta2 > theta1)
    if theta2 > 1:
        theta2 = 1
    if theta1 < 0:
        theta1 = 0

    # compute number of samples
    N = 1 / ((theta2 - theta1) ** 2) * math.log(1 / delta) * (math.sqrt(3 * theta1) + math.sqrt(2 * theta2)) ** 2
    N = math.ceil(N)

    global count
    count += N

    s = 0
    for i in range(N):
        if random.random() < p:
            s += 1
    mean = s / N
    if mean <= theta1 + (theta2 - theta1) * math.sqrt(3 * theta1) / (
            math.sqrt(3 * theta1) + math.sqrt(2 * theta2)):
        return 'Yes'
    else:
        return 'No'


# Our method
m = 0


def sequential(p0, p1, alpha, beta, p):
    dm = 0
    global m
    k1 = math.log(p1 / p0)
    k2 = math.log((1 - p1) / (1 - p0))
    Aln = math.log(1 / alpha)
    Bln = math.log(beta)
    while True:
        m += 1
        if random.random() < p:
            dm += 1
        fm = dm * k1 + (m - dm) * k2
        if fm > Aln:
            return 'Yes'
        elif fm < Bln:
            return 'No'


theta = 0.1  # p1 (lower bound)
eta = 0.01  # p0-p1 (interval)
delta = 0.01  # confidence
p = 0.5  # real probability

c_ours = 0
c_icse = 0
for i in range(100):
    count = 0
    m = 0
    sequential(p0=theta + eta, p1=theta, alpha=delta, beta=delta, p=p)
    adaptive_assert(theta=theta, eta=eta, delta=delta, obj_to_sample='xxx', timeout=300, p=p)
    c_ours += m
    c_icse += count
print('Ours:', c_ours / 100)
print('PROVERO:', c_icse / 100)
