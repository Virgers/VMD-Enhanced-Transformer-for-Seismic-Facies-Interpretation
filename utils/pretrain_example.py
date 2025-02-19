import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

# Generate random data
def generate_random_data(batch_size, image_size):
    images = torch.rand((batch_size, 1, image_size, image_size))
    masks = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()
    return images, masks

# Training loop
def train_unet(model, device, num_epochs=10, batch_size=4, image_size=64):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        images, masks = generate_random_data(batch_size, image_size)
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save model weights
def save_model_weights(model, path="unet_weights.pth"):
    torch.save(model.state_dict(), path)

# Load model weights
def load_model_weights(model, path="unet_weights.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    print('Model loaded successfully.')

class DownstreamModel(nn.Module):
    def __init__(self, pretrained_unet, projection_out_channels=1):
        super(DownstreamModel, self).__init__()
        self.pretrained_unet = pretrained_unet
        # Freeze the pre-trained U-Net if needed
        for param in self.pretrained_unet.parameters():
            param.requires_grad = False  # Freeze the pre-trained model
        
        # Add a new projection layer
        self.projection_layer = nn.Conv2d(
            in_channels=1,  # Output channels of U-Net
            out_channels=projection_out_channels,
            kernel_size=1
        )

    def forward(self, x):
        # Pass through the pre-trained U-Net
        features = self.pretrained_unet(x)
        # Pass through the projection layer
        output = self.projection_layer(features)
        return torch.sigmoid(output)
    
def generate_downstream_data(batch_size, image_size):
    images = torch.rand((batch_size, 1, image_size, image_size))
    labels = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()
    return images, labels

# Training loop for downstream task
def train_downstream(model, device, num_epochs=10, batch_size=4, image_size=64):
    model.to(device)
    criterion = nn.BCELoss()  # Use appropriate loss for your task
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        images, labels = generate_downstream_data(batch_size, image_size)
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Main function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    
    # Train the model
    train_unet(model, device)
    
    # Save the model weights
    save_model_weights(model)
    
    # Load the model weights
    load_model_weights(model)
    
    print("Model training and weight saving/loading completed.")

    # as for we train the down streaming model
    # Step 1: Load the pre-trained U-Net model
    pretrained_unet = UNet()
    load_model_weights(pretrained_unet, path="unet_weights.pth")
    
    # Step 2: Create the downstream model with an additional projection layer
    downstream_model = DownstreamModel(pretrained_unet, projection_out_channels=1)
    
    # Step 3: Train the downstream model
    train_downstream(downstream_model, device)
    
    print("Downstream task training completed.")
