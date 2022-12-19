import hashlib

import matplotlib.pyplot as plt

from IPython import display


def file_hash(file):
    hasher = hashlib.blake2b()
    with open(file, 'rb') as fd:
        while chunk := fd.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def plot(scores, mean_scores, rolling_mean_scores):
    _ = display.clear_output(wait=True)
    _ = display.display(plt.gcf())
    _ = plt.clf()
    _ = plt.title('Training...')
    _ = plt.xlabel('Number of Games')
    _ = plt.ylabel('Score')
    _ = plt.plot(scores)
    _ = plt.plot(mean_scores)
    _ = plt.plot(rolling_mean_scores)
    _ = plt.ylim(ymin=0)
    _ = plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    _ = plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    _ = plt.text(len(rolling_mean_scores)-1, rolling_mean_scores[-1], str(rolling_mean_scores[-1]))
    _ = plt.show()
    return
