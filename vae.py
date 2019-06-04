from torch import nn
from torch.autograd import Variable

class VAE(nn.Module) :

    def __init__(self, input_dim, zdims, h_num):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.h_num = h_num
        self.zdims = zdims
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        
        # h_num = 1
        if self.h_num == 1 :
            # Encoder
            self.h1dim = round((self.input_dim+self.zdims) * (1/2))
            self.fc1 = nn.Linear(input_dim, self.h1dim)
            self.fc21 = nn.Linear(self.h1dim, self.zdims) # mu
            self.fc22 = nn.Linear(self.h1dim, self.zdims) # log_var   
            # Decoder
            self.fc3 = nn.Linear(self.zdims, self.h1dim)
            self.fc4 = nn.Linear(self.h1dim, self.input_dim) # from latent space to output
            
        # h_num = 2 :
        if self.h_num == 2 :
            # Encoder
            self.h1dim = round((self.input_dim+self.zdims) * (2/3))
            self.h2dim = round((self.input_dim+self.zdims) * (1/3))
            self.fc1 = nn.Linear(input_dim, self.h1dim)
            self.fc2 = nn.Linear(self.h1dim, self.h2dim)
            self.fc31 = nn.Linear(self.h2dim, self.zdims)
            self.fc32 = nn.Linear(self.h2dim, self.zdims)
            # Decoder
            self.fc4 = nn.Linear(self.zdims, self.h2dim)
            self.fc5 = nn.Linear(self.h2dim, self.h1dim)
            self.fc6 = nn.Linear(self.h1dim, self.input_dim)
        
        
    def encode(self, x) :
        if self.h_num == 1 :
            h1 = self.leakyrelu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1) # returns mu, log_var
        if self.h_num == 2 :
            h1 = self.leakyrelu(self.fc1(x))
            h2 = self.leakyrelu(self.fc2(h1))
            return self.fc31(h2), self.fc32(h2)
    
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
        if self.h_num == 1 :
            h3 = self.leakyrelu(self.fc3(z))
            return self.sigmoid(self.fc4(h3)) # final output, reconstruction of mnist image
        if self.h_num == 2 :
            h4 = self.leakyrelu(self.fc4(z))
            h5 = self.leakyrelu(self.fc5(h4))
            return self.sigmoid(self.fc6(h5))
    
    def forward(self, x) :
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    