import torch
from torchvision.models import resnet18, ResNet18_Weights, ResNet
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import time
from tqdm import tqdm

class DeepSVDD:
    def __init__(self, objective, nu:float, c=None, R=0.0, optimizer_name:str="adam", lr:float=0.2, n_epoch:int=10,
                 lr_milestones:tuple=(), batch_size:int=128, weight_decay:float=1e-6, device:str="cpu", n_jobs_dataloader:int=0,
                 net_name:str="resnet"):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        assert objective in ("one-class", "soft-boundary"), "Invalid objective: one-class or soft-boundary"
        self.objective = objective
        self.R = torch.tensor(R, device=self.device)
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.test_auc = None
        self.train_time = None
        self.epoch_losses = []
        self.net = self.load_network(net_name)
        self.rep_dim = self.get_rep_dim(net_name)
        

    def train(self, dataset):
        self.net = self.net.to(device=self.device)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs_dataloader)
        optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name=="amsgrad")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones, gamma=0.1)

        train_start_time = time.time()
        
        if self.c is None:
            self.c = self.init_c_center(train_loader)
        
        self.net.train()
        for epoch in range(self.n_epoch):
            loss_epoch = 0.0
            n_batches = 0
            for batch_input, batch_label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epoch}"):
                batch_input = batch_input.to(device=self.device)
                optimizer.zero_grad(set_to_none=True)
                output = self.net(batch_input)
                dist = torch.sum((output - self.c)**2, dim=1)
                if self.objective == "soft-boundary":
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                elif self.objective == "one-class":
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                if self.objective == "soft-boundary":
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()
            print(f"Epoch {epoch+1}/{self.n_epoch}    : Loss {loss_epoch/n_batches}")
            self.epoch_losses.append(loss_epoch/n_batches)
        train_end_time = time.time()
        self.train_time = (train_end_time - train_start_time) // 60

        
    def test(self, dataset):
        self.net = self.net.to(self.device)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs_dataloader)
        test_labels, test_scores = [], []

        self.net.eval()
        with torch.no_grad():
            for batch_input, batch_label in test_loader:
                batch_input = batch_input.to(device=self.device)
                batch_label
                output = self.net(batch_input)
                dist = torch.sum((output - self.c)**2, dim=1)
                if self.objective == "soft-boundary":
                    scores = dist - self.R**2
                elif self.objective == "one-class":
                    scores = dist
                
                batch_label_lst = batch_label.cpu().data.numpy().tolist()
                scores_lst = scores.cpu().data.numpy().tolist()

                for l, s in zip(batch_label_lst, scores_lst):
                    test_labels.append(l)
                    test_scores.append(s)
        
        test_labels = np.array(test_labels)
        test_scores = np.array(test_scores)

        self.test_auc = roc_auc_score(test_labels, test_scores)

    def init_c_center(self, train_loader, eps=0.1):
        n_samples = 0
        c = torch.zeros(self.rep_dim, device=self.device)

        self.net = self.net.to(device=self.device)
        self.net.eval()
        with torch.no_grad():
            for batch_input, batch_label in train_loader:
                batch_input = batch_input.to(self.device)
                output = self.net(batch_input)
                n_samples += batch_input.shape[0]
                c += torch.sum(output, dim=0)
                break
        c = c / n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_rep_dim(self, net_name):
        if net_name == "resnet":
            dummy = torch.randn(1,3,224,224, device=self.device)
            self.net = self.net.to(self.device)
            output = self.net(dummy)
            rep_dim = output.shape[1]
            return rep_dim
        else:
            raise NotImplementedError
        
    def load_network(self, net_name) -> ResNet:
        if net_name == "resnet":
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
            net.fc = nn.Identity()
            for name, module in net.named_modules():
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = False
            return net
        else:
            raise NotImplementedError

def get_radius(dist: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)






