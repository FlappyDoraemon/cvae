require 'nn'
require 'cunn'
require 'image'
VAE = require 'VAE'
require 'nngraph'
require 'optim'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'
local process={}
local encoder
local decoder

local function convertCifar100BinToTorchTensor(inputFname, outputFname)
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamples = length / 3074 -- 1 coarse-label byte, 1 fine-label byte, 3072 pixel bytes

    assert(nSamples == math.floor(nSamples), 'expecting numSamples to be an exact integer')
    m:seek(1)

    local coarse = torch.ByteTensor(nSamples)
    local fine = torch.ByteTensor(nSamples)
    local data = torch.ByteTensor(nSamples, 3, 32, 32)
    for i=1,nSamples do
        coarse[i] = m:readByte()
        fine[i]   = m:readByte()
        local store = m:readByte(3072)
        data[i]:copy(torch.ByteTensor(store))
    end

    local out = {}
    out.data = data
    out.label = fine
    out.labelCoarse = coarse
    print(out)
    torch.save(outputFname, out)
end

function process.train(iteration,times)
    
--  loading the dataset file

    if(not paths.filep("cifar100-train.t7")) then
        os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz')
        os.execute('tar -xvf cifar-100-binary.tar.gz')
        convertCifar100BinToTorchTensor('cifar-100-binary/train.bin', 'cifar100-train.t7')
        convertCifar100BinToTorchTensor('cifar-100-binary/test.bin', 'cifar100-test.t7')
    end
    trainset = torch.load('cifar100-train.t7')
    setmetatable(trainset, 
        {__index = function(t, i) 
                       return {t.data[i], t.label[i]} 
                   end}
    );
    -- convert the data from a ByteTensor to a DoubleTensor.
    local trainset_cuda_pre = trainset.data:float()/255.0
    local trainset_inputcuda = trainset_cuda_pre:cuda()

--  define the network and the training settings

    local input_size = trainset.data:size(3)      -- 32 for c100
    local latent_variable_size = 128
    local encoder
    local decoder
    local reconstruction, model
    local z
    local mean, log_var
    local input

--  load or set up the model

    if(not paths.filep("cvae_encoder.t7")) then
        encoder = VAE.get_encoder(input_size, latent_variable_size)
        encoder = encoder:cuda()
        decoder = VAE.get_decoder(input_size, latent_variable_size)
        decoder = decoder:cuda()
        input = nn.Identity()()
        mean, log_var = encoder(input):split(2)
        z = nn.Sampler()({mean, log_var})
        reconstruction = decoder(z)
        model = nn.gModule({input},{reconstruction, mean, log_var})   
        -- model = mdoel:cuda()    
    else
	encoder = torch.load('cvae_encoder.t7')
        encoder = encoder:cuda()
        decoder = torch.load('cvae_decoder.t7')
        decoder = decoder:cuda()
        input = nn.Identity()()
        mean, log_var = encoder(input):split(2)
        z = nn.Sampler()({mean, log_var})
        reconstruction = decoder(z)
        model = nn.gModule({input},{reconstruction, mean, log_var})
        -- model = mdoel:cuda()
    end 
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false
    criterion = criterion:cuda()
    KLD = nn.KLDCriterion()
    KLD = KLD:cuda()

--  training

    local parameters, gradients = model:getParameters()
    local config = {
        learningRate = 0.001
    }
    local state = {}
    local lowerbound = 0
    local index_lenth = trainset.data:size(1)
    for i = 1 , times do
        print('main iteration:',iteration,'; train_function iteration:',i,'/',times,)
        local tic = torch.tic()
        for j = 1 , index_lenth do
            local seq = torch.randperm(index_lenth)  
            local inputs = trainset_inputcuda[seq[i]]              -- here maybe more suitable to be input, without end-s

            local opfunc = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end
                model:zeroGradParameters()
                local reconstruction, reconstruction_var, mean, log_var
                reconstruction, mean, log_var = unpack(model:forward(inputs))
                local err = criterion:forward(reconstruction, inputs)
                local df_dw = criterion:backward(reconstruction, inputs)

                local KLDerr = KLD:forward(mean, log_var)
                local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))
                error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
                model:backward(inputs, error_grads)                                                 -- how to understand this?
                local batchlowerbound = err + KLDerr

                return batchlowerbound, gradients
            end

            x, batchlowerbound = optim.adam(opfunc, parameters, config, state)
            lowerbound = lowerbound + batchlowerbound[1]
        end
        print("Main Epoch:" .. iteration .. "Epoch: " .. i .. " Lowerbound: " .. lowerbound/index_lenth .. " time: " .. torch.toc(tic)) 
    end
    torch.save('cvae_encoder.t7',encoder)
    torch.save('cvae_decoder.t7',decoder)
end

function process.view(img_temp)
    local encoder
    local decoder
    local reconstruction, model
    local mean, log_var
    local input = img_temp:float()/255.0
    encoder = torch.load('cvae_encoder.t7')
    decoder = torch.load('cvae_decoder.t7')
    input = nn.Identity()()
    mean, log_var = encoder(input):split(2)
    reconstruction = decoder(mean)
    model = nn.gModule({input},{reconstruction, mean, log_var})
    local reconstruction, reconstruction_var, mean, log_var
    reconstruction, mean, log_var = unpack(model:forward(inputs))
    local names = torch.load('c100_names.t7')
    image.display{image = img_temp*255.0,zoom=4,legend=names.fine_label_names[outputnum]}
    image.display{image = reconstruction*255.0,zoom=4,legend='reconstructed'}
end


return process


