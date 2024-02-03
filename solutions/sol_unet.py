class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """ Encoder """
        self.e1 = encoder(3, 64)
        self.e2 = encoder(64, 128)
        self.e3 = encoder(128, 256)
        self.e4 = encoder(256, 512)

        """ Bottleneck """
        self.bott = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder(1024, 512)
        self.d2 = decoder(512, 256)
        self.d3 = decoder(256, 128)
        self.d4 = decoder(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        
        """ Encoder """
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.bott(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        """ Classifier """
        outputs = self.outputs(d4)

        return outputs