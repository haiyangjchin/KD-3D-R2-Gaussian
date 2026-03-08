import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """3D double convolution block with batch normalization and ReLU."""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownSample3D(nn.Module):
    """Downsampling block with max pooling and double convolution."""
    
    def __init__(self, in_channels, out_channels):
        super(DownSample3D, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)


class UpSample3D(nn.Module):
    """Upsampling block with transposed convolution and double convolution."""
    
    def __init__(self, in_channels, out_channels):
        super(UpSample3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x, skip_connection):
        x = self.up(x)
        # Ensure dimensions match for skip connection
        diff_depth = skip_connection.size()[2] - x.size()[2]
        diff_height = skip_connection.size()[3] - x.size()[3]
        diff_width = skip_connection.size()[4] - x.size()[4]
        
        # Pad if necessary
        x = F.pad(x, [diff_width // 2, diff_width - diff_width // 2,
                      diff_height // 2, diff_height - diff_height // 2,
                      diff_depth // 2, diff_depth - diff_depth // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class CTUNet3D(nn.Module):
    """3D U-Net for CT Reconstruction from multiple 2D projections.
    
    Args:
        in_channels (int): Number of input channels (usually 1 when treating angles as depth)
        out_channels (int): Number of output channels (usually 1 for density volume)
        features (list): List of feature map sizes for each level
        use_dropout (bool): Whether to use dropout in the bottleneck
        target_depth (int): Target depth dimension for output volume (default: None, same as input depth)
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], use_dropout=False, target_depth=None):
        super(CTUNet3D, self).__init__()
        self.target_depth = target_depth
        
        # Encoder path
        self.encoder1 = DoubleConv3D(in_channels, features[0])
        self.encoder2 = DownSample3D(features[0], features[1])
        self.encoder3 = DownSample3D(features[1], features[2])
        self.encoder4 = DownSample3D(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = DownSample3D(features[3], features[3] * 2)
        
        # Decoder path
        self.decoder4 = UpSample3D(features[3] * 2, features[3])
        self.decoder3 = UpSample3D(features[3], features[2])
        self.decoder2 = UpSample3D(features[2], features[1])
        self.decoder1 = UpSample3D(features[1], features[0])
        
        # Final output layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.5) if use_dropout else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, in_channels, depth, height, width)
               where depth dimension represents projection angles.
        
        Returns:
            Reconstructed 3D volume of shape (batch, out_channels, depth_out, height_out, width_out)
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        if self.dropout is not None:
            bottleneck = self.dropout(bottleneck)
        
        # Decoder with skip connections
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        
        # Final output
        output = self.final_conv(dec1)
        
        # Adjust depth dimension if target_depth is specified
        if self.target_depth is not None and output.size(2) != self.target_depth:
            # Use trilinear interpolation to adjust depth
            output = F.interpolate(output, size=(self.target_depth, output.size(3), output.size(4)), mode='trilinear', align_corners=False)
        
        # Apply sigmoid activation to get density values in [0, 1] range
        output = torch.sigmoid(output)
        
        return output


class ProjectionEncoder(nn.Module):
    """Alternative architecture: Encode each 2D projection separately then fuse.
    
    This is an alternative approach that processes each projection independently
    with a 2D CNN, then fuses features to reconstruct 3D volume.
    """
    
    def __init__(self, num_projections, proj_height, proj_width, volume_depth, volume_height, volume_width):
        super(ProjectionEncoder, self).__init__()
        # Simplified version - to be implemented if needed
        pass


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a test input: batch_size=2, 1 channel, 32 angles, 64x64 projection size
    test_input = torch.randn(2, 1, 32, 64, 64).to(device)
    
    # Create model
    model = CTUNet3D(in_channels=1, out_channels=1, features=[16, 32, 64, 128]).to(device)
    
    # Forward pass
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Expected output shape should be (2, 1, 32, 64, 64) - same spatial dimensions
    # The depth dimension might change due to pooling/upsampling - adjust architecture if needed