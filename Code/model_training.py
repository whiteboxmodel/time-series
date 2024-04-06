import torch
import matplotlib.pyplot as plt


def train_epoch(xy_train, model, loss_fn, optimizer):
    model.train(mode = True) # switch to the training mode to calculate the gradients
    total_loss = 0.0
    for x, y in xy_train:
        model.zero_grad()
        y_hat = model(x) # predict with the current model
        loss = loss_fn(y_hat, y) # calculate the loss (error) for the prediction
        loss.backward() # backpropagation - calculates gradients
        optimizer.step() # update parameters to reduce the loss
        total_loss += loss.item()
    total_loss /= len(xy_train)
    return total_loss


def test_epoch(xy_test, model, loss_fn):
    model.train(mode = False) # switch to evaluation mode, do not calculate gradients
    total_loss = 0.0
    for x, y in xy_test:
        y_hat = model(x) # predict with the current model on the test data
        loss = loss_fn(y_hat, y) # calculate the loss (error) for the prediction
        total_loss += loss.item()
    total_loss /= len(xy_test)
    return total_loss


def train_model(xy_train, xy_test, model, loss_fn, optimizer, n_epochs):
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        train_loss = train_epoch(xy_train, model, loss_fn, optimizer)
        test_loss = test_epoch(xy_test, model, loss_fn)
        # Keep track of the train and test losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_loss, test_loss = round(train_loss, 3), round(test_loss, 3)
        print(f'Epoch {epoch}, train loss - {train_loss}, test loss - {test_loss}')
    return train_losses, test_losses


def plot_loss_history(train_losses, val_losses, start_epoch = 0):
    x_range = range(start_epoch, len(train_losses))
    plt.plot(x_range, train_losses[start_epoch:], label = 'train_loss', color = 'blue')
    plt.plot(x_range, val_losses[start_epoch:], label = 'val_loss', color = 'green')
    plt.legend()
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
