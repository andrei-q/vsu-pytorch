import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import copy
import math
import os
import progressbar
import numpy as np

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 64
learning_rate = 0.001

# Dataset
DATA_PATH = '/data/mnist'
MODEL_STORE_PATH = 'saved_models'

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
data_parts = ['train', 'valid']
mnist_datasets = dict()
mnist_datasets['train'] = torchvision.datasets.MNIST(
	root=DATA_PATH, train=True, transform=trans, download=False)
mnist_datasets['valid'] = torchvision.datasets.MNIST(
	root=DATA_PATH, train=False, transform=trans, download=False)

dataloaders = {p: DataLoader(mnist_datasets[p], batch_size=batch_size,
		shuffle=True, num_workers=4) for p in data_parts}

dataset_sizes = {p: len(mnist_datasets[p]) for p in data_parts}
num_batch = dict()
num_batch['train'] = math.ceil(dataset_sizes['train'] / batch_size)
num_batch['valid'] = math.ceil(dataset_sizes['valid'] / batch_size)
print(num_batch)


class NeuralNetworkModel(nn.Module):  # Net
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 3, kernel_size=5,	stride=1, padding=2)
		self.fc1 = nn.Linear(7*7*3, 10)

	def forward(self, x):
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2) 
		x = x.view(-1, 7*7*3)
		x = self.fc1(x)
		return F.log_softmax(x, dim=1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))

		# Each epoch has a training and validation phase
		for phase in data_parts:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			bar = progressbar.ProgressBar(maxval=num_batch[phase]).start()

			for i_batch, (inputs, labels) in enumerate(dataloaders[phase]):

				inputs = inputs.to(device)
				labels = labels.to(device)

				bar.update(i_batch)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):

					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				#print('epoch {} [{}]: {}/{}'.format(epoch, phase, i_batch, num_batch[phase]))
				#print('preds: ', preds)
				#print('labels:', labels.data)
				#print('match: ', int(torch.sum(preds == labels.data)))

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)			 

			bar.finish()   

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('Epoch {} [{}]: loss={:.4f}, acc={:.4f}' .
				format(epoch, phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'valid' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	#time_elapsed = time.time() - since
	#print('Training complete in {:.0f}m {:.0f}s'.format(
	#	time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


if __name__ == '__main__':

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")

	# Initialize model
	model = NeuralNetworkModel()	
	model = model.to(device)

	# Print model's state_dict
	print("Model's state_dict:")
	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	# Initialize optimizer
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	# Print optimizer's state_dict
	print("Optimizer's state_dict:")
	for var_name in optimizer.state_dict():
		print(var_name, "\t", optimizer.state_dict()[var_name])

	criterion = nn.CrossEntropyLoss()
	model = train_model(model, criterion, optimizer, exp_lr_scheduler,
	num_epochs=30)			

	#PATH = 'saved/model'
	#torch.save(model, PATH)  