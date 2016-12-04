--script source: https://github.com/torch/tutorials/blob/master/2_supervised/4_train.lua
--script tailored for country classficcation

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'hdf5'
require 'nn'


-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Geoprediction Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-input_img_h5','data/train/data.h5','path to the h5file containing the image feature')
   cmd:option('-input_labels_h5','data/train/labels.h5','path to labels for input images ')
   cmd:option('-save', 'results/country/', 'subdirectory to save/log experiments in')
  --  cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end

print '==>load training data'
local h5_img_file = hdf5.open(opt.input_img_h5, 'r')
local h5_label_file = hdf5.open(opt.input_labels_h5, 'r')

local data = h5_img_file:read('/image-features'):all()
local labels = h5_label_file:read('/country_labels'):all()

h5_img_file:close()
h5_label_file:close()

print '==> defining some tools'
--list classes
classes = {}
num_classes = 152
trsize = 11126
ninputs = 4096
nhiddens1 = 4096
nhiddens2 = 2048
for i=0,num_classes do
  table.insert(classes, i)
end


print '==>defining the model'
-- Simple 2-layer neural network, with tanh hidden units
 model = nn.Sequential()
 -- model:add(nn.Reshape(ninputs))
 model:add(nn.Linear(ninputs,nhiddens1))
 model:add(nn.Tanh())
 model:add(nn.Linear(nhiddens1,nhiddens2))
 model:add(nn.Tanh())
 model:add(nn.Linear(nhiddens2,num_classes))


print '==>defining the criterion'
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------

print '==>Train the model'
-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the model
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

--for now lets choose the optimization as SGD
print '==> configuring optimizer'
optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd


print '==> defining training procedure'

function train()

  --epoch tracker
--for epoch = 1,e3 do
      epoch = epoch or 1
      --local vars
      local time = sys.clock()

      --shuffle at each epoch
      shuffle = torch.randperm(trsize)

      -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trsize,opt.batchSize do
      -- create mini batch
          local inputs = {}
          local targets = {}
          for i = t,math.min(t+opt.batchSize-1,trsize) do
             -- load new sample
             local input = data[shuffle[i]]

             local target = labels[shuffle[i]]

	     local total_correct = 0

             if opt.type == 'double' then
                input = input:double()
                if opt.loss == 'mse' then
                   target = target:double()
                end
             elseif opt.type == 'cuda' then
                input = input:cuda();
                if opt.loss == 'mse' then
                   target = target:cuda()
                end
             end
             table.insert(inputs, input)
             table.insert(targets, target)
          end

          -- create closure to evaluate f(X) and df/dX
          local feval = function(x)
                           -- get new parameters
                           if x ~= parameters then
                              parameters:copy(x)
                           end

                           -- reset gradients
                           gradParameters:zero()

                           -- f is the average of all criterions
                           local f = 0

                           -- evaluate function for complete mini batch
                           for i = 1,#inputs do
                              -- estimate f

                              local output = model:forward(inputs[i])
			      pred, idx = output:max(1)
			      print(type(idx:number()))
			      print(type(targets[i]))
			      if idx==targets[i] then
			      	total_correct = total_correct + 1
				print(total_correct)
			     end 
                              local err = criterion:forward(output, targets[i])
                              --print(err);
                              f = f + err

                              -- estimate df/dW
                              local df_do = criterion:backward(output, targets[i])
                              model:backward(inputs[i], df_do)

                              -- -- update confusion
                              -- confusion:add(output, targets[i])
                           end

                           -- normalize gradients and f(X)
                           gradParameters:div(#inputs)
                           f = f/#inputs

                           -- return f and df/dX
                           return f,gradParameters
                        end

          -- optimize on current mini-batch
          if optimMethod == optim.asgd then
             _,_,average = optimMethod(feval, parameters, optimState)
          else
             optimMethod(feval, parameters, optimState)
          end
       end

       -- time taken
       time = sys.clock() - time
       time = time / trsize
       print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

      --  -- print confusion matrix
      --  print(confusion)

       -- save/log current net
       local filename = paths.concat(opt.save, 'model.net')
       os.execute('mkdir -p ' .. sys.dirname(filename))
       print('==> saving model to '..filename)
       torch.save(filename, model)

      --  -- next epoch
      --  confusion:zero()
       epoch = epoch + 1
    end
--end

while true do
   train()
  --  test()
end
