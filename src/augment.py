import random

def spec_augment(spec):
    spec = spec.clone()

    if random.random() < 0.5:
        t = random.randint(0, 100)
        t0 = random.randint(0, spec.shape[0] - t)
        spec[t0:t0+t, :] = 0

    if random.random() < 0.5:
        f = random.randint(0, 20)
        f0 = random.randint(0, spec.shape[1] - f)
        spec[:, f0:f0+f] = 0

    return spec