import torch  # Import PyTorch for model handling
from torch.utils.data import DataLoader  # Import DataLoader for batching dataset
from models.yolo import Model  # Import YOLO model class from Securade.ai HUB (from SingleInference_YOLOV7)
from utils.datasets import LoadImagesAndLabels  # Import dataset loader from YOLOv7 repository
import os  # Import os for file path handling
import yaml  # Import YAML for configuration handling

# Define paths and parameters
model_path = 'modelzoo/safety.pt'  # Path to the original safety.pt file
cfg_path = 'configs/custom_yolov7.yaml'  # Path to updated YAML configuration
data_path = 'data.yaml'  # Path to dataset configuration
epochs = 10  # Number of training epochs (adjust based on dataset size)
batch_size = 16  # Batch size (adjust based on GPU memory)
img_size = 640  # Image size (matches safety.pt and SingleInference_YOLOV7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
learning_rate = 0.01  # Learning rate for optimizer
weight_decay = 5e-4  # Weight decay for regularization
new_nc = 12  # Updated number of classes (from 10 to 12)

# Load YAML configuration to get anchors
with open(cfg_path, 'r') as file:
    yaml_data = yaml.safe_load(file)
class_names = yaml_data['names']

# Calculate number of anchors per grid from YAML
anchors = yaml_data.get('anchors', [])
num_anchors = len(anchors[0]) // 2 if anchors else 3  # Assume 3 if undefined

try:
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")  # Check if safety.pt exists

    # Load the original model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # Load safety.pt, map to CPU/GPU
    original_nc = checkpoint['model'].yaml['nc']  # Get original number of classes
    original_names = checkpoint['model'].names  # Get original class names
    print(f"Original number of classes: {original_nc}")  # Print original nc
    print(f"Original class names: {original_names}")  # Print original class names

    # Load model
    model = Model(cfg=cfg_path, nc=new_nc).to(device)

    # Load state dictionary, ignoring keys for layer size mismatches
    state_dict = checkpoint['model'].state_dict().copy()

    # Remove layers with mismatched sizes from state_dict
    out_channels = (new_nc + 5) * num_anchors
    mismatched_keys =[]
    for k, v in state_dict.items():
        # Common YOLOv7 "Detect" head keys for output convs will look like "model.77.m.0.weight"
        if k.startswith('model.77.m.') and v.shape[0] != out_channels:
            mismatched_keys.append(k)
    for k in mismatched_keys:
        print(f"Removing mismatched key from checkpoint: {k} {state_dict[k].shape}")
        state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)  # Load only compatible keys
    model.names = class_names  # Set or override names
    print(f"New number of classes: {model.yaml['nc']}")  # Confirm new nc
    print(f"New class names: {model.names}")  # Print new class names

    # Ensure detection layers are correct for the new class count
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Confirm it is the output layer for classes
            expected_out_channels = (new_nc + 5) * num_anchors
            if module.out_channels == expected_out_channels:
                torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
    # Reinitialize detection layer weights for new number of classes
    for name, param in model.named_parameters():
        if 'model.77.m' in name:
            if param.ndim == 4:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
            elif param.ndim == 1:
                torch.nn.init.constant_(param, 0.0)

    
    # Prepare dataset
    dataset = LoadImagesAndLabels('/Users/charoensupthawornt/Work/Securade.ai/hub/images/train', img_size=img_size)  # Load training images and labels
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Create DataLoader with 4 workers for faster loading

    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # SGD optimizer with standard YOLO hyperparameters

    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(epochs):  # Loop over epochs
        total_loss = 0  # Track total loss for the epoch
        for images, targets in dataloader:  # Loop over batches
            images, targets = images.to(device), targets.to(device)  # Move data to device
            optimizer.zero_grad()  # Clear previous gradients
            loss = model(images, targets)[1].sum()  # Compute YOLO loss (sum of box, objectness, class losses)
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update model weights
            total_loss += loss.item()  # Accumulate loss
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(dataloader)}')  # Print average loss per epoch

    # Save the fine-tuned model
    save_path = 'modelzoo/custom_safety.pt'  # Path for new model
    torch.save({
        'model': model,  # Save model object
        'epoch': epochs,  # Save epoch number
        'optimizer': optimizer.state_dict(),  # Save optimizer state
        'model_yaml': model.yaml  # Save updated YAML configuration
    }, save_path)  # Save to new file
    print(f"Saved fine-tuned model to {save_path}")  # Confirm save

except Exception as e:
    print(f"Error: {e}")  # Print any errors during execution