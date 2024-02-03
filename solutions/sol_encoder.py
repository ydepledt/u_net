class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()

        # We use the conv_block we've defined in the previous snippet
        self.conv = conv_block(in_channels, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)

        p = self.pool(x)

        return x, p