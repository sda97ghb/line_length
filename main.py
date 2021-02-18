import os

import matplotlib
matplotlib.use("Qt5Agg")

import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

plt.style.use('seaborn-whitegrid')


def is_package(module):
    return isinstance(module, str) and not module.endswith(".py") or hasattr(module, "__path__")


def package_path(module):
    return module if isinstance(module, str) else module.__path__[0]


def module_path(module):
    return module if isinstance(module, str) else module.__file__


def module_files(module):
    if is_package(module):
        for root, dirs, files in os.walk(package_path(module)):
            for filename in files:
                if filename.endswith('.py'):
                    yield os.path.join(root, filename)
    else:
        yield module_path(module)


def iter_lines(module):
    """Iterate over all lines of Python in module"""
    for filename in module_files(module):
        with open(filename) as f:
            yield from f


def lognorm_model(x, theta):
    amp, mu, sigma = theta
    return amp * stats.lognorm.pdf(x, scale=np.exp(mu), s=sigma)

def minfunc(theta, lengths, freqs):
    return np.sum((freqs - lognorm_model(lengths, theta)) ** 2)

def lognorm_mode(amp, mu, sigma):
    return np.exp(mu - sigma ** 2)

def lognorm_std(amp, mu, sigma):
    var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    return np.sqrt(var)


def hist_linelengths_with_fit(module, ax, indices=slice(50)):
    counts, bins, _ = hist_linelengths(module, ax)
    lengths = 0.5 * (bins[:-1] + bins[1:])
    opt = optimize.minimize(minfunc, x0=[1E5, 4, 0.5],
                            args=(lengths[indices], counts[indices]),
                            method='Nelder-Mead')
    model_counts = lognorm_model(lengths, opt.x)
    ax.fill_between(lengths, model_counts, alpha=0.3, color='gray')
    
    # Add text describing mu and sigma
    
    A, mu, sigma = opt.x
    mode = np.exp(mu - sigma ** 2)
    ax.text(0.22, 0.15, 'mode = {0:.1f}'.format(lognorm_mode(*opt.x)),
            transform=ax.transAxes, size=14)
    ax.text(0.22, 0.05, 'stdev = {0:.1f}'.format(lognorm_std(*opt.x)),
            transform=ax.transAxes, size=14)
    
    return opt.x


def make_module_title(module):
    if isinstance(module, str):
        return module
    return "{0} {1}".format(module.__name__, getattr(module, "__version__", ""))


def hist_linelengths(module, ax):
    """Plot a histogram of lengths of unique lines in the given module"""
    lengths = [len(line.rstrip('\n')) for line in set(iter_lines(module))]
    h = ax.hist(lengths, bins=np.arange(125) + 0.5, histtype='step', linewidth=1.5)
    ax.axvline(x=79.5, linestyle=':', color='black')
    ax.set(title=make_module_title(module),
           xlim=(1, 120),
           ylim=(0, None),
           xlabel='characters in line',
           ylabel='number of lines')
    return h


def main():
    import numpy, PyQt5, django, ariadne, gunicorn, decouple, django_prometheus, pika, tenacity, dramatiq, django_dramatiq, requests, feedgen, apscheduler
    modules = [numpy, PyQt5, django, ariadne, gunicorn, decouple, django_prometheus, pika, tenacity, dramatiq, django_dramatiq, requests, feedgen, apscheduler, "/home/dmitry/Projects/prototype/"]
    modules = [
        numpy, 
        PyQt5, 
        pika, 
        decouple, 
        tenacity, 
        
        requests, 
        ariadne, 
        apscheduler, 
        feedgen, 
        gunicorn, 
        
        django, 
        django_prometheus, 
        django_dramatiq, 
        dramatiq, 
        "/home/dmitry/Projects/prototype/"
    ]

    fig, ax = plt.subplots(3, 5, figsize=(14, 6), sharex=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for axi, module in zip(ax.flat, modules):
        hist_linelengths_with_fit(module, ax=axi)
        

    for axi in ax[0]:
        axi.set_xlabel('')
    for axi in ax[:, 1:].flat:
        axi.set_ylabel('')

    plt.show(block=True)


if __name__ == "__main__":
    main()
