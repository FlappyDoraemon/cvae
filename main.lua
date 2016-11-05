local train_process = require 'process'
require 'nn'

-------------------------------------------------------------------
-- PART 1 : Load Dataset
-------------------------------------------------------------------

local function convertSTL10BinToTorchTensor(inputFname, inputLabelsFname, outputFname)
    local nSamples = 0
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamplesF = length / (3*96*96) 
    assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
    nSamples = nSamples + nSamplesF
    m:close()

    local data = torch.ByteTensor(nSamples, 3, 96, 96)
    local index = 1
    local m=torch.DiskFile(inputFname, 'r'):binary()
    m:seekEnd()
    local length = m:position() - 1
    local nSamplesF = length / (3*96*96)
    m:seek(1)
    for j=1,nSamplesF do
        local store = m:readByte(3*96*96)
        data[index]:copy(torch.ByteTensor(store))
        index = index + 1
    end
    m:close()


    local out = {}
    
    out.data = data:transpose(3,4)

    if inputLabelsFname then
        local m=torch.DiskFile(inputLabelsFname, 'r'):binary()
        out.label = torch.ByteTensor(m:readByte(nSamplesF))
        m:close()
    end

    print(out)
    torch.save(outputFname, out)
end

if((not paths.filep('stl10-unlabeled.t7')) and (not paths.filep("stl10-unlabeled-scaled.t7"))) then
    os.execute('wget -c http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz')
    os.execute('tar -xvf stl10_binary.tar.gz')
    convertSTL10BinToTorchTensor('stl10_binary/unlabeled_X.bin',nil,'stl10-unlabeled.t7')
    convertSTL10BinToTorchTensor('stl10_binary/train_X.bin','stl10_binary/train_y.bin', 'stl10-train.t7')
    convertSTL10BinToTorchTensor('stl10_binary/test_X.bin','stl10_binary/test_y.bin', 'stl10-test.t7')
end

if(not paths.filep("stl10-unlabeled-scaled.t7")) then
    -- unlabeled data
    unlabeledset = torch.load('stl10-unlabeled.t7')   
    unlabeledlength = unlabeledset.data:size(1)
    local out  ={}
    out.data = torch.ByteTensor(unlabeledlength , 3 , 128 , 128)
    for i = 1 , unlabeledlength do
        out.data[i] = image.scale(unlabeledset.data[i],128,128) 
    end
    torch.save('stl10-unlabeled-scaled.t7', out)
    -- train data  
    trainset = torch.load('stl10-train.t7')   
    trainlength = trainset.data:size(1)
    local out  ={}
    out.data = torch.ByteTensor(trainlength , 3 , 128 , 128)
    out.label = trainset.label
    for i = 1 , trainlength do
        out.data[i] = image.scale(trainset.data[i],128,128) 
    end
    torch.save('stl10-train-scaled.t7', out)  

    -- test data  
    testset = torch.load('stl10-test.t7')   
    testlength = testset.data:size(1)
    local out  ={}
    out.data = torch.ByteTensor(testlength , 3 , 128 , 128)
    out.label = testset.label
    for i = 1 , testlength do
        out.data[i] = image.scale(testset.data[i],128,128) 
    end
    torch.save('stl10-test-scaled.t7', out)  
end

----------------------------------------------------------------------
-- PART 2 : Demo Procedure
----------------------------------------------------------------------
--  demo procedure

train_process.setmodel(128 , 512)
train_process.loadmodel()
train a around
train_process.train(2,100,true)
train_process.train(2,100,false)
train_process.savemodel()
train_process.loadmodel()
testset = torch.load('stl10-test-scaled.t7')
for i = 31 , 50 do
    train_process.view(testset.data[i])
end
train_process.mk_label_file(512)
train_process.loadlogis()
train_process.logis_train(1 , 100, 512)
train_process.savelogis()
