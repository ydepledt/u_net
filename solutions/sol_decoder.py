class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        
        # Here the upsampling is done by a transposed convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

        # Here we apply the conv_block we've defined in the previous snippet (think about dimensions)
        self.conv = conv_block(2*out_channels, out_channels)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)

        # Here we concatenate the output of the upsampling with the output of the encoder at the same level
        x = torch.cat([x, skip], axis=1)

        x = self.conv(x)
        
        return x