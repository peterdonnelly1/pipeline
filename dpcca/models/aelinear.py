"""=============================================================================
Linear autoencoder.
============================================================================="""

from   torch import nn

DEBUG=0
# ------------------------------------------------------------------------------

class AELinear(nn.Module):

    def __init__(self, cfg):
        """Initialize simple linear model.
        """
        
        print ( "AELINEAR:       INFO:     at \033[35;1m __init__()\033[m" )
        
        super(AELinear, self).__init__()
        
        self.input_dim = cfg.N_GENES
        emb_dim        = cfg.GENE_EMBED_DIM
        
        self.fc1       = nn.Linear(self.input_dim, emb_dim)
        self.fc2       = nn.Linear(emb_dim, self.input_dim)
   
        if DEBUG>0:
          print( "AELINEAR:       INFO:   __init__() \033[1m values are: self.input_dim=\033[35;1m{:}\033[m, emb_dime=\033[35;1m{:}\033[m, self.fc1=\033[35;1m{:}\033[m, self.fc2=\033[35;1m{:}\033[m"\
.format( self.input_dim, emb_dim, self.fc1, self.fc2 ) )
          print( "AELINEAR:       INFO:   __init__() MODEL dimensions: (input layer) m1 = \033[35;1m{:} x {:}\033[m; (output layer) m2 = \033[35;1m{:} x {:}\033[m"\
.format( self.input_dim, emb_dim, emb_dim, self.input_dim ) )
          print ("AELINEAR:       INFO:   __init__() \033[31;1mcaution: the gene input vectors must be the same dimensions as m1\033[m, i.e. \033[35;1m{:} x {:}\033[m".format( self.input_dim, emb_dim, emb_dim ) )

          print ("AELINEAR:       INFO:   __init__() \033[35;1mabout to return from AELinear()\033[m" )
        
# ------------------------------------------------------------------------------

    def encode(self, x):

        if DEBUG>0:
          print ( "AELINEAR:       INFO:                  encode(): x.shape           = {:}".format( x.shape ) ) 
          print ( "AELINEAR:       INFO:                  encode(): self.fc1(x).shape = {:}".format( (self.fc1(x)).shape ) )        
       

        return self.fc1(x)
        
# ------------------------------------------------------------------------------

    def decode(self, z):
        return self.fc2(z)

# ------------------------------------------------------------------------------

    def forward(self, x):
		
        if DEBUG>0:
          print ( "AELINEAR:       INFO:   forward(): batch_imagesr.shape = {:}".format( x1.shape ) )
          print ( "AELINEAR:       INFO:   forward(): batch_genesr.shape  = {:}".format( x2.shape ) )
          		
        z = self.encode(x.view(-1, self.input_dim))

        if DEBUG>0:
          print ( "AELINEAR:       INFO:   forward(): z1.shape = {:}".format( z1.shape ) )
          print ( "AELINEAR:       INFO:   forward(): z2.shape = {:}".format( z2.shape ) )
          
        return self.decode(z)
