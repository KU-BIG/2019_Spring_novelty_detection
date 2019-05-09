from torch import nn
from torch.autograd import Variable

ZDIMS=20

class VAE(nn.Module) : 
    def __init__(self, zdims):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        
        self.leakyrelu = nn.LeakyReLU()
        
        self.fc21 = nn.Linear(400, zdims) # mu
        self.fc22 = nn.Linear(400, zdims) # log_var
        
        
        # Decoder
        self.fc3 = nn.Linear(zdims, 400)
        
        self.fc4 = nn.Linear(400, 784) # from latent space to output
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x) :
        h1 = self.leakyrelu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1) # returns mu, log_var
    
    def reparametrize(self, mu, logvar) : 
        if self.training :
            std = logvar.mul(0.5).exp_() # log_var
            eps = Variable(std.data.new(std.size()).normal_())
            # for training, which enalbes backpropagation
            return eps.mul(std).add_(mu)
        
        else :
            # for inference, just use mu ?????
            return mu
        
    def decode(self, z) :
        h3 = self.leakyrelu(self.fc3(z))
        return self.sigmoid(self.fc4(h3)) # final output, reconstruction of mnist image
    
    def forward(self, x) :
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    