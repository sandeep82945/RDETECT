import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

class CustomDataset(Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)

  def __getitem__(self, index):
    return self.features[index], self.labels[index]

class MyNN(nn.Module):
  def __init__(self, input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
    super().__init__()
    layers = []

    for i in range(num_hidden_layers):
      layers.append(nn.Linear(input_dim, neurons_per_layer))
      layers.append(nn.BatchNorm1d(neurons_per_layer))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout_rate))
      input_dim = neurons_per_layer

    layers.append(nn.Linear(neurons_per_layer, output_dim))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

def NN_initializer(X, y, **options):
    batch_size = options['batch_size']
    input_dim = options['input_dim']
    output_dim = options['output_dim']
    num_hidden_layers = options['num_hidden_layers']
    neurons_per_layer = options['neurons_per_layer']
    dropout_rate = options['dropout_rate']
    learning_rate = options['learning_rate']
    weight_decay = options['weight_decay']
    optimizer_name = options['optimizer_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = MyNN(input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, criterion, optimizer, train_loader, test_loader, test_dataset

def NN_initializer1(**options):
    input_dim = options['input_dim']
    output_dim = options['output_dim']
    num_hidden_layers = options['num_hidden_layers']
    neurons_per_layer = options['neurons_per_layer']
    dropout_rate = options['dropout_rate']
    learning_rate = options['learning_rate']
    weight_decay = options['weight_decay']
    optimizer_name = options['optimizer_name']

    model = MyNN(input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, criterion, optimizer

def score_finder(true_vals, pred_vals):
    accuracy = accuracy_score(true_vals, pred_vals)
    precision = precision_score(true_vals, pred_vals, average='binary')
    recall = recall_score(true_vals, pred_vals, average='binary')
    f1 = f1_score(true_vals, pred_vals, average='binary')

    print(f"Accuracy : {accuracy}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    print(f"F1-score : {f1}")

def tester(model, test_data, device=device):
  model.eval()

  with torch.no_grad():
    X_test = test_data.features.to(device)
    logits = model(X_test)

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int()

    score_finder(test_data.labels, preds.detach().cpu())
   
# objective function
def objective(trial, X, y):
  # next hyperparameter values from the search space
  num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
  neurons_per_layer = trial.suggest_int("neurons_per_layer", 128, 1024, step=128)
  epochs = trial.suggest_int("epochs", 50, 1000, step=50)
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
  dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
  batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
  optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop'])
  weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

  # model init
  input_dim = 1000
  output_dim = 1

  model, criterion, optimizer, train_loader, test_loader, test_dataset = NN_initializer(
    X,
    y,
    batch_size=batch_size ,
    num_hidden_layers= num_hidden_layers,
    neurons_per_layer= neurons_per_layer,
    dropout_rate=dropout_rate ,
    learning_rate= learning_rate,
    weight_decay= weight_decay,
    optimizer_name= optimizer_name,
    input_dim = input_dim,
    output_dim = output_dim
  )

  # training loop
  for epoch in range(epochs):
    for batch_features, batch_labels in train_loader:
      batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

      outputs = model(batch_features)
      loss = criterion(outputs, batch_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # evaluation
  model.eval()

  with torch.no_grad():
    f1 = 0
    for batch_features, batch_labels in test_loader:
      batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
      logits = model(batch_features)

      probs = torch.sigmoid(logits)
      preds = (probs >= 0.5).int()
      f1 += f1_score(batch_labels.detach().cpu(), preds.detach().cpu())

  return f1

def model_trainer(ai,human, model_params):
    input_dim = model_params['input_dim']
    output_dim = model_params['output_dim']
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    
    X = {id:ai[id]+human[id] for id in list(ai.keys())}
    y = {id:[[1] for _ in range(len(ai[id]))] + [[0] for _ in range(len(human[id]))] for id in list(ai.keys())}

    data = {id:train_test_split(X[id], y[id], test_size=0.2) for id in list(X.keys())}
    X_train = []
    y_train = []
    for id in list(data.keys()):
        X_train += data[id][0]
        y_train += data[id][2]

    test_data = {id:{'X':data[id][1], 'y':data[id][3]} for id in list(data.keys())}

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model, criterion, optimizer= NN_initializer1(
        **model_params,
    )

    # training loop
    for epoch in tqdm(range(epochs)):
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Testing process
    for id in list(test_data.keys()):
       X_test = torch.tensor(test_data[id]['X'], dtype=torch.float32)
       y_test = torch.tensor(test_data[id]['y'], dtype=torch.float32)
       test_dataset = CustomDataset(X_test, y_test)
       tester(model,test_dataset)

    return model
