local train_process = require 'process'
require 'nn'

-------------------------------------------------------------------
-- PART 0 : CIFAR TEST
-------------------------------------------------------------------

for idx = 1 , 10 do
    train_process.train(idx, 90)
end

if(not paths.filep("cifar100-train.t7")) then
    os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz')
    os.execute('tar -xvf cifar-100-binary.tar.gz')
    convertCifar100BinToTorchTensor('cifar-100-binary/train.bin', 'cifar100-train.t7')
    convertCifar100BinToTorchTensor('cifar-100-binary/test.bin', 'cifar100-test.t7')
end
testset = torch.load('cifar100-test.t7')
trainset = torch.load('cifar100-train.t7')

for i = 1 , 10 do
    train_process.view(testset.data[i],i)
end


----------------------------------------------------------------------
-- PART 1 : Remain for use
----------------------------------------------------------------------

--[[
if(not paths.filep("cifar100-train.t7")) then
        os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz')
        os.execute('tar -xvf cifar-100-binary.tar.gz')
        convertCifar100BinToTorchTensor('cifar-100-binary/train.bin', 'cifar100-train.t7')
        convertCifar100BinToTorchTensor('cifar-100-binary/test.bin', 'cifar100-test.t7')
    end
    testset = torch.load('cifar100-test.t7')
    trainset = torch.load('cifar100-train.t7')
    local i
    local mark = 0
    local lenth = testset.data:size(1)
    image.display{image = testset.data[1],zoom = 5}
--]]

--[[
local input = frame:float()/255.0
local output = net:forward(input:cuda())
local currentmax = output[{1}]
local i
local outputnum = 0
for j = 1 , 99 do
    if output[{j+1}] > currentmax then
        outputnum = j
        currentmax = output[{j+1}]
    end
end
     
--]]

