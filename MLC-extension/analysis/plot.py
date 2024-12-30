import numpy as np
import matplotlib.pyplot as plt


def plot_train_history(train_tracker):
    n_episode = len(train_tracker)
    train_loss = []
    val_loss = []
    for i, episode in enumerate(train_tracker):
        train_loss.append(episode['avg_train_loss'])
        val_loss.append(episode['val_loss'])
    
        
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    fig, ax = plt.subplots()
    ax.plot(range(n_episode), train_loss, label='train_loss')
    ax.plot(range(n_episode), val_loss, label='val_loss')
    ax.legend()
    plt.show()
    a=1
