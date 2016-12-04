require 'torch'
require 'optim'
require 'nn'
require 'image'

-- We will write the loss to a text file and read from there to plot the loss as training proceeds
logger = optim.Logger('loss_log.txt')

-----------------------------------------------------------------------
-- 1. Load data

-- local trsize = 2217
-- local img_size = 32
-- local train_file = 'data/train/train_coordinates.t7'
-- trainData = torch.load(train_file)
dofile 'data.lua'

data = trainData.data
coordinates = trainData.coordinates
data = data:double()
coordinates = coordinates:double()
-- data = data:reshape(trsize,3,img_size,img_size)
-- data = data:float()

----------------------------------------------------------------------
-- 2. Define the model (predictor)
--one output for each latitude and longitude
    noutputs = 2
    -- input dimensions
    nfeats = 3
    width = 32
    height = 32
    ninputs = nfeats*width*height

    -- hidden units, filter sizes
    nstates = {64,64,128}
    filtsize = 5
    poolsize = 2
    normkernel = image.gaussian1D(7)

     model = nn.Sequential()

     -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
     model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
     model:add(nn.Tanh())
     model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
     model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

     -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
     model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
     model:add(nn.Tanh())
     model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
     model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

     -- stage 3 : standard 2-layer neural network
     model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
     model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
     model:add(nn.Tanh())
     model:add(nn.Linear(nstates[3], noutputs))


----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.


criterion = nn.MSECriterion()


----------------------------------------------------------------------
-- 4. Train the model

x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our model, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end
   local inputs = data[_nidx_]
   local target = coordinates[_nidx_]
  --  local target = sample[{ {1} }]      -- this funny looking syntax allows
  --  local inputs = sample[{ {2,3} }]    -- slicing of arrays.

   -- reset gradients (gradients are always accumulated, to accommodate
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.

-- we cycle 1e4 times over our training data
for i = 1,1e4 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#data)[1] do

      -- optim contains several optimization algorithms.
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x,
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific

      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#data)[1]
   print('current loss = ' .. current_loss)

   logger:add{['training error'] = current_loss}
   logger:style{['training error'] = '-'}
   logger:plot()
end

----------------------------------------------------------------------
-- 5. Test the trained model.

-- Now that the model is trained, one can test it by evaluating it
-- on new samples.

-- The text solves the model exactly using matrix techniques and determines
-- that
--   corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides

-- We compare our approximate results with the text's results.

-- text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}
--
-- print('id  approx   text')
-- for i = 1,(#data)[1] do
--    local myPrediction = model:forward(data[i][{{2,3}}])
--    print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
-- end
