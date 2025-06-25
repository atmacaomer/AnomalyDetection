from sklearn.ensemble import IsolationForest
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import time
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model_name:str, n_epochs=20, lr=0.2, weight_decay=1e-6, batch_size=128, device:str="cuda",
                 n_jobs_dataloader:int=0, optimizer_name:str="adam", lr_milestones:tuple=()):
        self.model = None
        if model_name == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.optimizer_name = optimizer_name
        self.lr_milestones = lr_milestones
    def fit(self, dataset:ImageFolder):
        num_classes = len(dataset.classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs_dataloader)
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=self.optimizer_name)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones, gamma=0.1)
        cross_entropy = nn.CrossEntropyLoss()        
        self.model.train()
        for epoch in range(self.n_epochs):
            loss_epoch = 0
            n_batches = 0
            for batch_input, batch_label in tqdm(train_loader, desc=f"Epoch {epoch+1} / {self.n_epochs}"):
                batch_input = batch_input.to(self.device)
                batch_label = batch_label.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(batch_input)
                loss = cross_entropy(output, batch_label)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
                      
            print(f"Epoch {epoch+1}/{self.n_epochs}    : Loss{loss_epoch / n_batches}")
            scheduler.step()
    def get_feature_rep(self, dataset):
        features = []
        labels = []
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs_dataloader)
        self.model.eval()
        original_fc = self.model.fc
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            print("Getting the features..")
            for batch_input, batch_label in loader:
                batch_input = batch_input.to(self.device)
                output = self.model(batch_input)
                features.append(output.cpu())
                labels.append(batch_label.cpu())
            print("Getting the features has finished.")
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        self.model.fc = original_fc
        return features, labels
    
class DeepIsolationForest:
    def __init__(self, model_name:str, n_epochs=20, lr=0.2, weight_decay=1e-6, batch_size=128, device:str="cuda",
                 n_jobs_dataloader:int=0, optimizer_name:str="adam", lr_milestones:tuple=()):
        self.model_trainer = ModelTrainer(model_name,n_epochs,lr,weight_decay,batch_size,
                                          device,n_jobs_dataloader,optimizer_name,lr_milestones)
        self.isolation_forest = IsolationForest(contamination="auto", random_state=42)
        self.test_auc = None
        self.train_time_model = None
        self.train_time_forest = None
    
    def fit(self, dataset):
        train_model_start_time = time.time()
        self.model_trainer.fit(dataset)
        train_model_end_time = time.time()
        self.train_time_model = (train_model_end_time - train_model_start_time) // 60
        train_forest_start_time = time.time()
        features, labels = self.model_trainer.get_feature_rep(dataset)
        self.isolation_forest.fit(X=features)
        train_forest_end_time = time.time()
        self.train_time_forest = (train_forest_end_time - train_forest_start_time) // 60

    def predict(self, dataset):
        features, labels = self.model_trainer.get_feature_rep(dataset)
        scores = -1 * self.isolation_forest.decision_function(X=features)
        self.test_auc = roc_auc_score(labels, scores)

    


        

    