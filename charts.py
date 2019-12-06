import matplotlib.pyplot as plt
import analysis

def histogram(seq, bins=10, title='Histogram'):
    plt.hist(seq, edgecolor = 'black', bins = bins)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def plot_comparison(model, batch, qty=96):
    comparison = analysis.comparison_matrix(model, batch, qty)
    plt.figure(figsize=(12,8))
    plt.imshow(comparison)
    plt.colorbar()