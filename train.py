# train.py
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import argparse
import json

# Define command-line arguments
parser = argparse.ArgumentParser(description='Parser for training script')
parser.add_argument('data_dir', help='Input data directory', type=str, default='flowers')
parser.add_argument('--save_dir', help='Input saving directory. Optional', type=str)
parser.add_argument('--arch', help='Model architecture: "alexnet" or "vgg13"', type=str, default='alexnet')
parser.add_argument('--learning_r', help='Learning rate (default: 0.001)', type=float, default=0.001)
parser.add_argument('--hidden_units', help='Number of hidden units (default: 512)', type=int, default=512)
parser.add_argument('--epochs', help='Number of epochs (default: 5)', type=int, default=5)
parser.add_argument('--gpu', help='Use GPU for training if available (default: False)', action='store_true')

args = parser.parse_args()

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

# Define data directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define data transforms for training, validation, and testing
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Load class-to-name mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load pretrained model and freeze parameters
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    num_features = model.classifier[0].in_features
else:
    model = models.alexnet(pretrained=True)
    num_features = model.classifier[1].in_features

for param in model.parameters():
    param.requires_grad = False

# Define custom classifier
classifier = nn.Sequential(
    nn.Linear(num_features, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(args.hidden_units, len(cat_to_name)),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier
model.to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_r)

# Train the classifier
epochs = args.epochs
print_every = 40
steps = 0

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print training loss and accuracy
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                accuracy = 0
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model.forward(inputs)
                    batch_loss = criterion(outputs, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

            running_loss = 0
            model.train()

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {
    'classifier': model.classifier,
    'input_size': num_features,
    'output_size': len(cat_to_name),
    'hidden_units': args.hidden_units,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epochs,
    'model_name': args.arch
}

if args.save_dir:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')
