import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = [8, 16, 32, 64, 128, 256]
    mious = [65.1, 65.5, 65.9, 66.1, 66.7, 66.8]
    oas = [89.3, 89.5, 89.7, 89.7, 89.7, 89.7]
    maccs = [71.5, 71.8, 72.3, 72.9, 73.3, 73.5]
    
    plt.plot(x, mious)