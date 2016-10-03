require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(iamge_edge, latent_variable_size)
     -- The Encoder
    local encoder = nn.Sequential()
    encoder:add(nn.Reshape(3, iamge_edge, iamge_edge))  
    encoder:add(nn.SpatialConvolution(3, 16, 4, 4, 2, 2, 1, 1))          --  #kernel = 32  edge = (32 - 4.. + 1.. * 2 + 2) / 2.. = 16
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialConvolution(16, 32, 4, 4, 2, 2, 1, 1))         --  #kernel = 32  edge = (16 - 4.. + 1.. * 2 + 2) / 2.. = 8
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialConvolution(32, 16, 4, 4, 2, 2, 1, 1))         --  #kernel = 16  edge = (8 - 4.. + 1.. * 2 + 2) / 2.. = 4
    encoder:add(nn.ReLU(true))
    encoder:add(nn.Reshape(4*4*16)) 
    
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(4*4*16, latent_variable_size))
    mean_logvar:add(nn.Linear(4*4*16, latent_variable_size))

    encoder:add(mean_logvar)
    
    return encoder
end

function VAE.get_decoder(iamge_edge , latent_variable_size)
    -- The Decoder
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, 4*4*16))
    decoder:add(nn.Reshape(16,4,4))                                       -- 16 * 4 * 4
    decoder:add(nn.SpatialFullConvolution(16, 32, 4, 4, 2, 2, 1, 1))         -- 32 * 8 * 8
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(32, 16, 4, 4, 2, 2, 1, 1))         -- 16 * 16 * 16
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(16, 3, 4, 4, 2, 2, 1, 1))          -- 3 * 32 * 32
    decoder:add(nn.Sigmoid(true))

    return decoder
end

return VAE
