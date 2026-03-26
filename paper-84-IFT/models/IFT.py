from layers.IFT_EncDec import *


class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        # RevIN
        self.revin = RevIN(self.CFG)

        # Embedding
        self.embedding = Embedding(self.CFG)

        # Encoder
        self.encoder = Encoder(self.CFG)

        # Forecaster
        self.forecaster = ImplicitForecaster(self.CFG)

    def forward(self, x, x_mark, y, y_mark):
        # Norm
        x_norm = self.revin(x, mode='norm')

        # Embed
        embedding = self.embedding(x_norm, x_mark)

        # Encode
        enc_out = self.encoder(embedding)

        # Forecast
        x = self.forecaster(enc_out, x)

        # Denorm
        x_denorm = self.revin(x, mode='denorm')

        return x_denorm
