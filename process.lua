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
local logis_model

function process.train(times,batch_size,inverse)
    
--  loading the dataset file
    --  cutorch.setDevice(3)
    encoder = encoder:cuda()
    decoder = decoder:cuda()
    print('loading dataset to memory, please wait...')
    trainset = torch.load('stl10-unlabeled-scaled.t7')

    setmetatable(trainset, 
        {__index = function(t, i) 
                       return {t.data[i], t.label[i]} 
                   end}
    );
    print('setting dataset metatable, please wait...')
    local input_size = trainset.data:size(3)   
    -- How many batch iterations we are going to train on
    local index_lenth = math.ceil(trainset.data:size(1) / batch_size)
    --  local trainset_float = torch.zeros(trainset.data:size(1) / train_set_split,3,input_size,input_size)
    --  trainset_float:copy(trainset.data[{{(train_set_split_idx-1)*index_lenth*batch_size+1 , train_set_split_idx*index_lenth*batch_size},{},{},{}}]:float()/255.0)

--  define the network and the training settings

    local reconstruction, model
    local z
    local mean, log_var
    local input

--  set up the model

    input = nn.Identity()()
    mean, log_var = encoder(input):split(2)
    z = nn.Sampler()({mean, log_var})
    reconstruction = decoder(z)
    model = nn.gModule({input},{reconstruction, mean, log_var})   
    --  model = model:cuda()
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
    for i = 1 , times do
        print('we will train the whole training set for the ' .. i .. ' th time. Batch size is ' .. batch_size )
        local lowerbound = 0
        --  get the shuffle sequence for this iteration
        local shuffle_seq = torch.randperm(trainset.data:size(1))  
        model:training()
        local tic = torch.tic()
        --  train on the whole unlabled set with the minibatch
        for j = 1 , index_lenth do   
        --  idnex_lenth: how many batches? go through it!
            local inputs = torch.zeros(batch_size, 3, input_size, input_size)
            for k = 1 , batch_size do
                --  inputs[k] = trainset_float[shuffle_seq[(j-1)*batch_size+k]]
                local img_temp = trainset.data[shuffle_seq[(j-1)*batch_size+k]]:float()/255.0
                if inverse == true then
                    inputs[k] = img_temp:index(3 ,torch.linspace(img_temp:size(3),1,img_temp:size(3)):long())
                else
                    inputs[k] = img_temp
                end
            end
            inputs = inputs:cuda()
            --  define the local training funvtion; need to ask JX for further knowledge
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
            lowerbound = lowerbound + batchlowerbound[1]/index_lenth/batch_size
        end
--[[
        --  evaluate on the test set
        model:evaluate()
        for t = 1,testData:size() do
            -- disp progress
            xlua.progress(t, testData:size())

            -- get new sample
            local input = testData.data[t]
            if opt.type == 'double' then input = input:double()
            elseif opt.type == 'cuda' then input = input:cuda() end
            local target = testData.labels[t]

            -- test sample
            local pred = model:forward(input)
            confusion:add(pred, target)
        end
        for j = 1 , 
--]]
        print("Main Epoch: " .. i .. '/' .. times .. "; Batch Iterations: " .. index_lenth .. "; Lowerbound: " .. lowerbound .. " time: " .. torch.toc(tic)) 
    end
end

function process.view(img_temp)
    --  cutorch.setDevice(3)
    local reconstruction, model
    local latent
    local inputs = img_temp:float()/255.0
------
    local reconstruction_image, mean, log_var
    encoder = encoder:float()
    decoder = decoder:float()
    latent = encoder:forward(inputs)--:split(2)
    reconstruction_image = decoder:forward(latent[1])
    image.display{image = torch.cat( img_temp:float(),reconstruction_image:float()*255.0,3), zoom=2}
end

function process.loadmodel()
    --  cutorch.setDevice(3)
    encoder = torch.load('cvae_encoder.t7')
    decoder = torch.load('cvae_decoder.t7')
    encoder = encoder:cuda()
    decoder = decoder:cuda()
    print('loaded previously saved VAE model successfully.')
end

function process.setmodel(input_size , latent_variable_size)
    --  cutorch.setDevice(3)
    print('setting the model')
    encoder = VAE.get_encoder(input_size, latent_variable_size)
    encoder = encoder:cuda()
    decoder = VAE.get_decoder(input_size, latent_variable_size)
    decoder = decoder:cuda()
    print('mdoel established')
end

function process.savemodel()
    --  cutorch.setDevice(3)
    parameters, gradParameters = encoder:getParameters()
    -- this need to be after :cuda and after :getParameters
    local lightModel = encoder:float():clone('weight','bias','running_mean','running_std')
    -- do training, and when need to save, call
    torch.save('cvae_encoder.t7',lightModel)
    encoder = nil
    lightmodel = nil
--    
    parameters, gradParameters = decoder:getParameters()
    -- this need to be after :cuda and after :getParameters
    local lightModel = decoder:float():clone('weight','bias','running_mean','running_std')
    -- do training, and when need to save, call
    torch.save('cvae_decoder.t7',lightModel)
    decoder = nil
    lightmodel = nil
    print('saved the VAE model successfully.')
end

function process.set_logis_model(latent_variable_size , label_size)
    if(not paths.filep("logis_model.t7")) then
        logis_model = nn.Sequential()
        logis_model:add(nn.Linear(latent_variable_size, label_size)) 
        logis_model:add(nn.LogSoftMax()) 
        logis_model = logis_model:cuda()
    end
end

function process.savelogis()
    --  cutorch.setDevice(3)
    logis_model = logis_model:float()
    parameters, gradParameters = logis_model:getParameters()
    -- this need to be after :cuda and after :getParameters
    local lightModel = logis_model:clone('weight','bias','running_mean','running_std')
    -- do training, and when need to save, call
    torch.save('logis_latent.t7',lightModel)
    logis_model = nil
    lightmodel = nil
    print('saved logix model successfully')
end

function process.loadlogis()
    --  cutorch.setDevice(3)
    logis_model = torch.load('logis_latent.t7')
    print('loaded previously saved logis model successfully.')
end

function process.mk_label_file(latent_variable_size)
    if(not paths.filep('stl10-train-scaled-latent_codes')) then

--  initialization

        local latent
        encoder = encoder:float()

--  train set

        local trainset = torch.load('stl10-train-scaled.t7')
        setmetatable(trainset, 
            {__index = function(t, i) 
                           return {t.data[i], t.label[i]} 
                       end}
        );
        local index_lenth = trainset.data:size(1)
        local codes = torch.zeros(index_lenth , latent_variable_size)
        --  local labels = torch.zeros(index_lenth , 10)
        local percent = 1
        print('Forwarding train set')
        for i = 1 , index_lenth do
            local img_input = trainset.data[i]:float()/255.0
            local latent_code = encoder:forward(img_input)--:split(2)
            codes[i] = latent_code[1]
            --  labels[{{i},{trainset.label[i]}}] = 1
            if i * 10 >= percent * index_lenth then
                print(percent .. '/10 of the whole train set forwarded.')
                percent = percent + 1
            end
        end
        torch.save('stl10-train-scaled-latent_codes.t7', codes) 
        --  torch.save('stl10-train-scaled-labels', labels)  

--  test set

        local testset = torch.load('stl10-test-scaled.t7')
        setmetatable(testset, 
            {__index = function(t, i) 
                           return {t.data[i], t.label[i]} 
                       end}
        );
        index_lenth = testset.data:size(1)
        codes = torch.zeros(index_lenth , latent_variable_size)
        --  labels = torch.zeros(index_lenth , 10)
        percent = 1
        print('Forwarding test set')
        for i = 1 , index_lenth do
            local img_input = testset.data[i]:float()/255.0
            local latent_code = encoder:forward(img_input)--:split(2)
            codes[i] = latent_code[1]
            --  labels[{{i},{trainset.label[i]}}] = 1
            if i * 10 >= percent * index_lenth then
                print(percent .. '/10 of the whole test set forwarded.')
                percent = percent + 1
            end
        end
        torch.save('stl10-test-scaled-latent_codes.t7', codes)
        --  torch.save('stl10-test-scaled-labels', labels)  
    end
end    

function process.logis_train(times , batch_size, latent_variable_size)
    logis_model = logis_model:cuda()
    local train_latent_set = torch.load('stl10-train-scaled-latent_codes.t7')
    local trainset = torch.load('stl10-train-scaled.t7')
    local index_lenth = math.ceil(train_latent_set:size(1) / batch_size)
    local criterion = nn.MSECriterion()
    criterion = criterion:cuda()
-- train some batch-turns
    for i = 1 , times do
        local percent = 1
        local seq = torch.randperm(train_latent_set:size(1))
        for j = 1 , index_lenth do
            local inputs = torch.zeros(batch_size, latent_variable_size)
            local outputs = torch.zeros(batch_size, 10)
            for k = 1 , batch_size do
                inputs[k] = train_latent_set[seq[(j-1)*batch_size+k]]
                outputs[{{k},{trainset.label[seq[(j-1)*batch_size+k]]}}] = 1  --  originally the label mark is from 1 to 10
            end
            inputs = inputs:cuda()     
            outputs = outputs:cuda()       
            criterion:forward(logis_model:forward(inputs), outputs)
            logis_model:zeroGradParameters()
            logis_model:backward(inputs, criterion:backward(logis_model.output, outputs)) -- net.output
            logis_model:updateParameters(0.005)
            if j * 10 >= percent * index_lenth then
                print('time:' .. i .. '/' .. times .. ';' .. percent .. '/10 of training finished')
                percent = percent + 1
            end
        end
    end
-- evaluate
    logis_model = logis_model:float()
    local test_latent_set = torch.load('stl10-test-scaled-latent_codes.t7')
    local testset = torch.load('stl10-test-scaled.t7')
    local index_lenth = test_latent_set:size(1) 
    local correct_count = torch.zeros(10)
    local total_count = torch.zeros(10)
    local total_accuracy = 0
    for j = 1 , index_lenth do
        local output = logis_model:forward(test_latent_set[j]:float())
        local tempmax = output[1]
        local templabel = 1
        for k = 2 , 10 do
            if output[k] > tempmax then
                tempmax = output[k]
                templabel = k
            end
        end
        if templabel == testset.label[j] then
            total_accuracy = total_accuracy + 1
            correct_count[testset.label[j]] = correct_count[testset.label[j]] + 1
        end
        total_count[testset.label[j]] = total_count[testset.label[j]] + 1
    end
    print('Prediction accuracy rate is:' .. total_accuracy/index_lenth*100 .. '%')
    for i = 1 , 10 do
        print('correct/total of label' .. i .. ': ' .. correct_count[i] .. '/' .. total_count[i] .. ';accuracy rate is: ' .. correct_count[i]/total_count[i])
    end
end


return process


