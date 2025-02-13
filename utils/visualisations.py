import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('steps')
    plt.ylabel('average loss')
    plt.title('Loss')
    plt.show()