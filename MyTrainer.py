#!/usr/bin/env python
# coding: utf-8

# In[62]:

import torch
import torchvision

# class DeviceDataLoader():
#     """
#         This is a class that helps to move data to available device
#     """
#     def __init__(self, dl):
#         import imp
#         try:
#             imp.find_module('torch')
#             found = True
#         except:
#             found = False
        
#         if found is False: raise Exception('torch module not present')
#         self.dl = dl
        
    # def get_device(self):
    #     """Picks GPU if available or else CPU"""
    #     if torch.cuda.is_available():
    #         return torch.device('cuda')
    #     else: return torch.device('cpu')
#     #moving data to available device gpu or cpu
#     def to_device(self,data, device):
#         if isinstance(data, (list, tuple)):
#             return [to_device(x, device) for x in data ]
#         return data.to(device, non_blocking=True)
    
#     def __iter__(self):
#         for b in self.dl:
#             yield self.to_device(b, self.get_device())
#     def __len__(self):
#         return len(self.dl)
    
        

class SebTorchTrainer():
    """
        A helper class for training <<<<<<<< Development stage {Beta version} >>>>>>>>>>>>>>
    """
    def __init__(self, model):
        self.model = model
        import imp
        try:
            imp.find_module('torch')
            found = True
        except:
            found = False
        
        if found is False: raise Exception('torch module not present')
        
    def spliter(self, train_size, dataset):
        """
        This method helps to split data randomly, by passing a float of the percentage of the training set
        """
        train_size = float(train_size)
        from torch.utils.data import random_split
        self.dataset = dataset
        
        if float(train_size) >= 1.00:
            raise TypeError('Size of train must be less than 1.00 {} '.format(train_size))
        else:
            train = round(len(self.dataset)*train_size)
            validation = len(self.dataset) - train
            train_set, validation_Set = random_split(self.dataset, [ train, validation])
            
            return train_set, validation_Set
        
    def prepare(self, train_bs=None, valid_bs=None, shuf_train=False, shuf_test=False, shuf_valid=False, train=True, validation=True, valid_set= None,train_set= None, train_split=None, dataset=None):
        """
        This method helps to prepare daata by converting them to batches
        """
        from torch.utils.data import DataLoader
        
        if train_bs is None: train_bs = 3
        if valid_bs is None: valid_bs = 3   
        if test_bs is None: test_bs = 10
        
        if train_set is None or valid_set is None and type(train_split)== 'float':
            train_set,valid_set = self.spliter(train_split, dataset )
        else: raise Exception('Train_split is probably not a float')
            
        train_loader = DataLoader(dataset=train_set, batch_size=train_bs, shuffle=shuf_train)
        valid_loader = DataLoader(dataset=valid_set, batch_size=valid_bs, shuffle=shuf_valid)
        
        return train_loader, valid_loader
        
    
    def fit(self, epochs, train_batch, validation_batch, loss_func, metric, opt_func, lr):
        """
        This is a fitter method that trains the model
        """
        opt = opt_func(self.model.parameters(), lr=lr)
        
        # train_batch = DeviceDataLoader(train_batch)
        # validation_batch = DeviceDataLoader(validation_batch)

        if torch.cuda.is_available():
            train_batch.cuda()
            validation_batch.cuda()
        else:
            train_batch.cpu()
            validation_batch.cpu()
        
        train_losses, train_acces = [], []

        valid_losses , valid_acces = [], []
        
        for epoch in range(epochs):

            train_acc = 0.0
            train_loss = 0.0
            valid_acc = 0.0
            valid_loss = 0.0
            batch_train = 0.0
            batch_val = 0.0

            for xb, yb in train_batch:
                self.model.train()
                # make predictions
                preds = self.model(xb)
                # calculate the loss
                loss = loss_func(preds, yb)
                # Calculate the derivatives
                loss.backward()
                # update training parameters
                opt.step()
                # reset gradients to x=zero
                opt.zero_grad()
                train_loss += loss.item() * len(xb)
                # validation on training
                acc = metric(preds, yb)

                train_acc += acc 
                batch_train += len(xb)      

            # Validation accuracy
            with torch.no_grad():
                for xval, yval in validation_batch:
                    self.model.eval()
                    val_pred = self.model(xval)

                    val_loss = loss_func(val_pred, yval).item()

                    val_acc = metric(val_pred, yval)

                    valid_acc += val_acc 
                    valid_loss += val_loss * len(xval)
                    batch_val += len(xval)              

            #         calculate average metrics (losses, accuracies)
            train_loss = train_loss/batch_train
            valid_loss = valid_loss/batch_val
            train_acc = train_acc/len(train_batch)
            valid_acc = valid_acc/len(validation_batch)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)


            print(' Epoch [{}/{}], training_loss: {:.4f}, training_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format( epoch+1, epochs, train_loss, train_acc, valid_loss, valid_acc ))
        return train_losses, train_acces, valid_losses , valid_acces
    
    def evaluate(self, validation_batch, loss_func, valid_dl, metric):
        
        valid_acc = 0.0
        valid_loss = 0.0
       
        
        with torch.no_grad():
            for xval, yval in validation_batch:
                self.model.eval()
                val_pred = self.model(xval)

                val_loss = loss_func(val_pred, yval).item()

                val_acc = metric(val_pred, yval)

                valid_acc += val_acc 
                valid_loss += val_loss * len(xval)
                
           
            valid_loss = valid_loss/len(validation_batch)
            valid_acc = valid_acc/len(validation_batch)

        return {'Acc: {:.4f}, loss: {:.4f}'.format(valid_acc, valid_loss)}   
    
# This trainer helper was built in a rush, it is open for modifications and still only implementation for classification, regression perspective loading...... 

# Author: @Sebago
# Mail: humiditii45@gmail.com
# Github: github.com/Humiditii
                 


# In[ ]:




