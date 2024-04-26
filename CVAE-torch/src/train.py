import torch
from torch.nn import functional as F

def loss_cvae(out_img, img, mu, logvar, beta=0.5):
    kl_div = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    recons_loss = F.mse_loss(out_img, img)
    loss = (1-beta)*recons_loss + beta*kl_div
    return loss, recons_loss, kl_div

def train(model, train_loader, optimizer, device, beta=0.5):
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out_imgs, mu, logvar = model(imgs, labels)

        loss, _, _ = loss_cvae(out_imgs, imgs,
                                              mu, logvar, beta)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss/len(train_loader.dataset)

def test(model, valid_loader, device, beta=0.5):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out_imgs, mu, logvar = model(imgs, labels)
            loss, _, _ = loss_cvae(out_imgs, imgs,
                                                mu, logvar, beta)
            valid_loss += loss.item()
    return valid_loss/len(valid_loader.dataset)