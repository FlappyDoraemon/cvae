require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(iamge_edge, latent_variable_size)
     -- The Encoder
    local encoder = nn.Sequential()
    encoder:add(nn.Reshape(3, iamge_edge, iamge_edge)) 
    encoder:add(nn.SpatialConvolution(3, 256, 4, 4, 2, 2, 1, 1))         --  #kernel = 256  edge = (16 - 4.. + 1.. * 2 + 2) / 2.. = 64
    encoder:add(nn.ReLU(true)) 
    encoder:add(nn.SpatialConvolution(256, 256, 4, 4, 2, 2, 1, 1))         --  #kernel = 256  edge = (16 - 4.. + 1.. * 2 + 2) / 2.. = 32
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))          --  #kernel = 512  edge = (96 - 3 .. + 1.. * 2 + 2) / 2.. = 16
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialConvolution(512, 512, 4, 4, 2, 2, 1, 1))         --  #kernel = 512  edge = (16 - 4.. + 1.. * 2 + 2) / 2.. = 8
    encoder:add(nn.ReLU(true))
    encoder:add(nn.SpatialConvolution(512, 1024, 4, 4, 2, 2, 1, 1))         --  #kernel = 1024  edge = (8 - 4.. + 1.. * 2 + 2) / 2.. = 4
    encoder:add(nn.ReLU(true))
    encoder:add(nn.Reshape(4*4*1024)) 
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(4*4*1024, latent_variable_size))
    mean_logvar:add(nn.Linear(4*4*1024, latent_variable_size))

    encoder:add(mean_logvar)
    
    return encoder
end

function VAE.get_decoder(iamge_edge , latent_variable_size)
    -- The Decoder
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, 4*4*1024))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.Reshape(1024,4,4))                                          -- 1024 * 4 * 4
    decoder:add(nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1))         -- 512 * 8 * 8
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(512, 512, 4, 4, 2, 2, 1, 1))         -- 512 * 16 * 16
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))         -- 256 * 32 * 32
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(256, 256, 4, 4, 2, 2, 1, 1))         -- 256 * 64 * 64
    decoder:add(nn.ReLU(true))
    decoder:add(nn.SpatialFullConvolution(256, 3, 4, 4, 2, 2, 1, 1))          -- 3 * 128 * 128
    decoder:add(nn.Sigmoid(true))

    return decoder
end

return VAE
