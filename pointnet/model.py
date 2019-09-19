from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Functional
import time
import torch.optim as optim
from torch.nn.init import xavier_uniform, calculate_gain
import math
import copy
import numpy as np
import helpers.functions as mhf
import cnn_models.help_fun as cnn_hf
import quantization
import quantization.help_functions

USE_CUDA = torch.cuda.is_available()


class STN3d(nn.Module):
    def __init__(self, conv_widths = [64,128,1024],  fc_widths = [1024,512,256]):
        # conv_width :
        #
        super(STN3d, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class PointNetCls_small(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat



class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

def train_model(model, train_loader, test_loader, initial_learning_rate = 0.001, use_nesterov=True,
                initial_momentum=0.9, weight_decayL2=0.00022, epochs_to_train=100, print_every=500,
                learning_rate_style='generic', use_distillation_loss=False, teacher_model=None,
                quantizeWeights=False, numBits=8, grad_clipping_threshold=False, start_epoch=0,
                bucket_size=None, quantizationFunctionToUse='uniformLinearScaling',
                backprop_quantization_style='none', estimate_quant_grad_every=1, add_gradient_noise=False,
                ask_teacher_strategy=('always', None), quantize_first_and_last_layer=True,
                mix_with_differentiable_quantization=False):

    # backprop_quantization_style determines how to modify the gradients to take into account the
    # quantization function. Specifically, one can use 'none', where gradients are not modified,
    # 'truncated', where gradient values outside -1 and 1 are truncated to 0 (as per the paper
    # specified in the comments) and 'complicated', which is the temp name for my idea which is slow and complicated
    # to compute

    if use_distillation_loss is True and teacher_model is None:
        raise ValueError('To compute distillation loss you have to pass the teacher model')

    if teacher_model is not None:
        teacher_model.eval()

    learning_rate_style = learning_rate_style.lower()
    lr_scheduler = cnn_hf.LearningRateScheduler(initial_learning_rate, learning_rate_style)
    new_learning_rate = initial_learning_rate
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, nesterov=use_nesterov,
                          momentum=initial_momentum, weight_decay=weight_decayL2)
    startTime = time.time()

    pred_accuracy_epochs = []
    percentages_asked_teacher = []
    losses_epochs = []
    informationDict = {}
    last_loss_saved = float('inf')
    step_since_last_grad_quant_estimation = 1
    number_minibatches_per_epoch = len(train_loader)

    if quantizeWeights:
        quantizationFunctionToUse = quantizationFunctionToUse.lower()
        if backprop_quantization_style is None:
            backprop_quantization_style = 'none'
        backprop_quantization_style = backprop_quantization_style.lower()
        if quantizationFunctionToUse == 'uniformAbsMaxScaling'.lower():
            s = 2 ** (numBits - 1)
            type_of_scaling = 'absmax'
        elif quantizationFunctionToUse == 'uniformLinearScaling'.lower():
            s = 2 ** numBits
            type_of_scaling = 'linear'
        else:
            raise ValueError('The specified quantization function is not present')

        if backprop_quantization_style is None or backprop_quantization_style in ('none', 'truncated'):
            quantizeFunctions = lambda x: quantization.uniformQuantization(x, s,
                                                    type_of_scaling=type_of_scaling,
                                                    stochastic_rounding=False,
                                                    max_element=False,
                                                    subtract_mean=False,
                                                    modify_in_place=False, bucket_size=bucket_size)[0]

        elif backprop_quantization_style == 'complicated':
            quantizeFunctions = [quantization.uniformQuantization_variable(s, type_of_scaling=type_of_scaling,
                                                    stochastic_rounding=False,
                                                    max_element=False,
                                                    subtract_mean=False,
                                                    modify_in_place=False, bucket_size=bucket_size) \
                                 for _ in model.parameters()]
        else:
            raise ValueError('The specified backprop_quantization_style not recognized')

        num_parameters = sum(1 for _ in model.parameters())

        def quantize_weights_model(model):
            for idx, p in enumerate(model.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == num_parameters-1:
                        continue #don't quantize first and last layer
                if backprop_quantization_style == 'truncated':
                    p.data.clamp_(-1, 1)
                if backprop_quantization_style in ('none', 'truncated'):
                    p.data = quantizeFunctions(p.data)
                elif backprop_quantization_style == 'complicated':
                    p.data = quantizeFunctions[idx].forward(p.data)
                else:
                    raise ValueError

        def backward_quant_weights_model(model):
            if backprop_quantization_style == 'none':
                return

            for idx, p in enumerate(model.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == num_parameters-1:
                        continue #don't quantize first and last layer

                # Now some sort of backward. For the none style, we don't do anything.
                # for the truncated style, we just need to truncate the grad weights
                # as per the paper here: https://arxiv.org/pdf/1609.07061.pdf
                # if we are quantizing, I put gradient values above 1 to 0.
                # their case it not immediately applicable to ours, but let's try this out
                if backprop_quantization_style == 'truncated':
                    p.grad.data[p.data.abs() > 1] = 0
                elif backprop_quantization_style == 'complicated':
                    p.grad.data = quantizeFunctions[idx].backward(p.grad.data)

    if print_every > number_minibatches_per_epoch:
        print_every = number_minibatches_per_epoch // 2

    try:
        epoch = start_epoch
        for epoch in range(start_epoch, epochs_to_train+start_epoch):
            print("begin training")
            if USE_CUDA:
                print("USE_CUDA")
            if mix_with_differentiable_quantization:
                print('=== Starting Quantized Distillation epoch === ')
            model.train()
            print_loss_total = 0
            count_asked_teacher = 0
            count_asked_total = 0
            for idx_minibatch, data in enumerate(train_loader, start=1):

                if quantizeWeights:
                    if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                        # we save them because we only want to quantize weights to compute gradients,
                        # but keep using non-quantized weights during the algorithm
                        model_state_dict = model.state_dict()
                        quantize_weights_model(model)

                model.zero_grad()
                print_loss, curr_c_teach, curr_c_total = forward_and_backward(model, data, idx_minibatch, epoch,
                                            use_distillation_loss=use_distillation_loss,
                                            teacher_model=teacher_model,
                                            ask_teacher_strategy=ask_teacher_strategy,
                                            return_more_info=True)
                count_asked_teacher += curr_c_teach
                count_asked_total += curr_c_total

                #load the non-quantize weights and use them for the update. The quantized
                #weights are used only to get the quantized gradient
                if quantizeWeights:
                    if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                        model.load_state_dict(model_state_dict)
                        del model_state_dict #free memory

                if add_gradient_noise and not quantizeWeights:
                    cnn_hf.add_gradient_noise(model, idx_minibatch, epoch, number_minibatches_per_epoch)

                if grad_clipping_threshold is not False:
                    # gradient clipping
                    for p in model.parameters():
                        p.grad.data.clamp_(-grad_clipping_threshold, grad_clipping_threshold)

                if quantizeWeights:
                    if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                        backward_quant_weights_model(model)

                optimizer.step()

                if step_since_last_grad_quant_estimation >= estimate_quant_grad_every:
                    step_since_last_grad_quant_estimation = 0

                step_since_last_grad_quant_estimation += 1

                # print statistics
                print_loss_total += print_loss
                if (idx_minibatch) % print_every == 0:
                    last_loss_saved = print_loss_total / print_every
                    str_to_print = 'Time Elapsed: {}, [Start Epoch: {}, Epoch: {}, Minibatch: {}], loss: {:3f}'.format(
                        mhf.timeSince(startTime), start_epoch+1, epoch + 1, idx_minibatch, last_loss_saved)
                    if pred_accuracy_epochs:
                        str_to_print += ' Last prediction accuracy: {:2f}%'.format(pred_accuracy_epochs[-1]*100)
                    print(str_to_print)
                    print_loss_total = 0

            curr_percentages_asked_teacher = count_asked_teacher/count_asked_total if count_asked_total != 0 else 0
            percentages_asked_teacher.append(curr_percentages_asked_teacher)
            losses_epochs.append(last_loss_saved)
            curr_pred_accuracy = evaluateModel(model, test_loader, fastEvaluation=False)
            pred_accuracy_epochs.append(curr_pred_accuracy)
            print(' === Epoch: {} - prediction accuracy {:2f}% === '.format(epoch + 1, curr_pred_accuracy*100))

            if mix_with_differentiable_quantization and epoch != start_epoch + epochs_to_train - 1:
                print('=== Starting Differentiable Quantization epoch === ')
                #the diff quant step is not done at the last epoch, so we end on a quantized distillation epoch
                model_state_dict = optimize_quantization_points(model, train_loader, test_loader, new_learning_rate,
                                            initial_momentum=initial_momentum, epochs_to_train=1, print_every=print_every,
                                            use_nesterov=use_nesterov,
                                            learning_rate_style=learning_rate_style, numPointsPerTensor=2**numBits,
                                            assignBitsAutomatically=True, bucket_size=bucket_size,
                                            use_distillation_loss=True, initialize_method='quantiles',
                                            quantize_first_and_last_layer=quantize_first_and_last_layer)[0]
                model.load_state_dict(model_state_dict)
                del model_state_dict  # free memory
                losses_epochs.append(last_loss_saved)
                curr_pred_accuracy = evaluateModel(model, test_loader, fastEvaluation=False)
                pred_accuracy_epochs.append(curr_pred_accuracy)
                print(' === Epoch: {} - prediction accuracy {:2f}% === '.format(epoch + 1, curr_pred_accuracy * 100))


            #updating the learning rate
            new_learning_rate, stop_training = lr_scheduler.update_learning_rate(epoch, 1-curr_pred_accuracy)
            if stop_training is True:
                break
            for p in optimizer.param_groups:
                try:
                    p['lr'] = new_learning_rate
                except:pass

    except Exception as e:
        print('An exception occurred: {}\n. Training has been stopped after {} epochs.'.format(e, epoch))
        informationDict['errorFlag'] = True
        informationDict['numEpochsTrained'] = epoch-start_epoch

        return model, informationDict
    except KeyboardInterrupt:
        print('User stopped training after {} epochs'.format(epoch))
        informationDict['errorFlag'] = False
        informationDict['numEpochsTrained'] = epoch - start_epoch
    else:
        print('Finished Training in {} epochs'.format(epoch+1))
        informationDict['errorFlag'] = False
        informationDict['numEpochsTrained'] = epoch + 1 - start_epoch

    if quantizeWeights:
       quantize_weights_model(model)

    if mix_with_differentiable_quantization:
        informationDict['numEpochsTrained'] *= 2

    informationDict['percentages_asked_teacher'] = percentages_asked_teacher
    informationDict['predictionAccuracy'] = pred_accuracy_epochs
    informationDict['lossSaved'] = losses_epochs
    return model, informationDict

def optimize_quantization_points(modelToQuantize, train_loader, test_loader, initial_learning_rate = 1e-5,
                                 initial_momentum=0.9, epochs_to_train=30, print_every=500, use_nesterov=True,
                                 learning_rate_style='generic', numPointsPerTensor=16,
                                 assignBitsAutomatically=False, bucket_size=None,
                                 use_distillation_loss=True, initialize_method='quantiles',
                                 quantize_first_and_last_layer=True):

    print('Preparing training - pre processing tensors')


    numTensorsNetwork = sum(1 for _ in modelToQuantize.parameters())
    initialize_method = initialize_method.lower()
    if initialize_method not in ('quantiles', 'uniform'):
        raise ValueError('The initialization method must be either quantiles or uniform')

    if isinstance(numPointsPerTensor, int):
        numPointsPerTensor = [numPointsPerTensor] * numTensorsNetwork

    if len(numPointsPerTensor) != numTensorsNetwork:
        raise ValueError('numPointsPerTensor must be equal to the number of tensor in the network')

    if quantize_first_and_last_layer is False:
        numPointsPerTensor = numPointsPerTensor[1:-1]

    #same scaling function that is used inside nonUniformQUantization. It is important they are the same
    scalingFunction = quantization.ScalingFunction('linear', False, False, bucket_size, False)


    #if assigning bits automatically, use the 2-norm of the gradient to determine weights importance
    if assignBitsAutomatically:
        num_to_estimate_grad = 5
        modelToQuantize.zero_grad()
        for idx_minibatch, batch in enumerate(train_loader, start=1):
            cnn_hf.forward_and_backward(modelToQuantize, batch, idx_batch=idx_minibatch, epoch=0,
                                                     use_distillation_loss=False)
            if idx_minibatch >= num_to_estimate_grad:
                break

        #now we compute the 2-norm of the gradient for each parameter
        fisherInformation = []
        for idx, p in enumerate(modelToQuantize.parameters()):
            if quantize_first_and_last_layer is False:
                if idx == 0 or idx == numTensorsNetwork - 1:
                    continue
            fisherInformation.append((p.grad.data/num_to_estimate_grad).norm())

        #zero the grad we computed
        modelToQuantize.zero_grad()

        #now we use a simple linear proportion to assign bits
        #the minimum number of points is half what was given as input
        numPointsPerTensor = quantization.help_functions.assign_bits_automatically(fisherInformation,
                                                                                   numPointsPerTensor,
                                                                                   input_is_point=True)

    #initialize the points using the percentile function so as to make them all usable
    pointsPerTensor = []
    if initialize_method == 'quantiles':
        for idx, p in enumerate(modelToQuantize.parameters()):
            if quantize_first_and_last_layer is True:
                currPointsPerTensor = numPointsPerTensor[idx]
            else:
                if idx == 0 or idx == numTensorsNetwork - 1:
                    continue
                currPointsPerTensor = numPointsPerTensor[idx-1]
            initial_points = quantization.help_functions.initialize_quantization_points(p.data,
                                                                                        scalingFunction,
                                                                                        currPointsPerTensor)
            initial_points = Variable(initial_points, requires_grad=True)
            # do a dummy backprop so that the grad attribute is initialized. We need this because we call
            # the .backward() function manually later on (since pytorch can't assign variables to model
            # parameters)
            initial_points.sum().backward()
            pointsPerTensor.append(initial_points)
    elif initialize_method == 'uniform':
        for numPoint in numPointsPerTensor:
            initial_points = torch.FloatTensor([x/(numPoint-1) for x in range(numPoint)])
            if USE_CUDA: initial_points = initial_points.cuda()
            initial_points = Variable(initial_points, requires_grad=True)
            # do a dummy backprop so that the grad attribute is initialized. We need this because we call
            # the .backward() function manually later on (since pytorch can't assign variables to model
            # parameters)
            initial_points.sum().backward()
            pointsPerTensor.append(initial_points)
    else: raise ValueError

    #dealing with 0 momentum
    options_optimizer = {}
    if initial_momentum != 0: options_optimizer = {'momentum':initial_momentum, 'nesterov':use_nesterov}
    optimizer = optim.SGD(pointsPerTensor, lr=initial_learning_rate, **options_optimizer)

    lr_scheduler = cnn_hf.LearningRateScheduler(initial_learning_rate, learning_rate_style)
    startTime = time.time()

    pred_accuracy_epochs = []
    losses_epochs = []
    last_loss_saved = float('inf')
    number_minibatches_per_epoch = len(train_loader)

    if print_every > number_minibatches_per_epoch:
        print_every = number_minibatches_per_epoch // 2

    modelToQuantize.eval()
    quantizedModel = copy.deepcopy(modelToQuantize)
    epoch = 0

    quantizationFunctions = []
    for idx, p in enumerate(quantizedModel.parameters()):
        if quantize_first_and_last_layer is False:
            if idx == 0 or idx == numTensorsNetwork - 1:
                continue
        #efficient version of nonUniformQuantization
        quant_fun = quantization.nonUniformQuantization_variable(max_element=False, subtract_mean=False,
                                                                 modify_in_place=False, bucket_size=bucket_size,
                                                                 pre_process_tensors=True, tensor=p.data)

        quantizationFunctions.append(quant_fun)

    print('Pre processing done, training started')

    for epoch in range(epochs_to_train):
        quantizedModel.train()
        print_loss_total = 0
        for idx_minibatch, data in enumerate(train_loader, start=1):

            #zero the gradient of the parameters model
            quantizedModel.zero_grad()
            optimizer.zero_grad()

            #quantize the model parameters
            for idx, p_quantized in enumerate(quantizedModel.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == numTensorsNetwork - 1:
                        continue
                    currIdx = idx - 1
                else: currIdx = idx
                #efficient quantization
                p_quantized.data = quantizationFunctions[currIdx].forward(None, pointsPerTensor[currIdx].data)

            print_loss = cnn_hf.forward_and_backward(quantizedModel, data, idx_minibatch, epoch,
                                        use_distillation_loss=use_distillation_loss,
                                        teacher_model=modelToQuantize)

            #now get the gradient of the pointsPerTensor
            for idx, p in enumerate(quantizedModel.parameters()):
                if quantize_first_and_last_layer is False:
                    if idx == 0 or idx == numTensorsNetwork - 1:
                        continue
                    currIdx = idx - 1
                else: currIdx = idx
                pointsPerTensor[currIdx].grad.data = quantizationFunctions[currIdx].backward(p.grad.data)[1]

            optimizer.step()

            #after optimzer.step() we need to make sure that the points are still sorted. Implementation detail
            for points in pointsPerTensor:
                points.data = torch.sort(points.data)[0]

            # print statistics
            print_loss_total += print_loss
            if (idx_minibatch) % print_every == 0:
                last_loss_saved = print_loss_total / print_every
                str_to_print = 'Time Elapsed: {}, [Epoch: {}, Minibatch: {}], loss: {:3f}'.format(
                    mhf.timeSince(startTime), epoch + 1, idx_minibatch, last_loss_saved)
                if pred_accuracy_epochs:
                    str_to_print += '. Last prediction accuracy: {:2f}%'.format(pred_accuracy_epochs[-1] * 100)
                print(str_to_print)
                print_loss_total = 0

        losses_epochs.append(last_loss_saved)
        curr_pred_accuracy = evaluateModel(quantizedModel, test_loader, fastEvaluation=False)
        pred_accuracy_epochs.append(curr_pred_accuracy)
        print(' === Epoch: {} - prediction accuracy {:2f}% === '.format(epoch + 1, curr_pred_accuracy * 100))

        # updating the learning rate
        new_learning_rate, stop_training = lr_scheduler.update_learning_rate(epoch, 1 - curr_pred_accuracy)
        if stop_training is True:
            break
        for p in optimizer.param_groups:
            try:
                p['lr'] = new_learning_rate
            except:
                pass

    print('Finished Training in {} epochs'.format(epoch + 1))
    informationDict = {'predictionAccuracy': pred_accuracy_epochs,
                       'numEpochsTrained': epoch+1,
                       'lossSaved':losses_epochs}

    #IMPORTANT: When there are batch normalization layers, important information is contained
    #also in the running mean and runnin var values of the batch normalization layers. Since these are not
    #parameters, they don't show up in model.parameter() list (and they don't have quantization points
    #associated with it). So if I return just the optimized quantization points, and quantize the model
    #weight with them, I will have inferior performance because the running mean and var of the batch normalization
    #layers won't be saved. To solve this issue I also return the quantized model state dict, that contains
    #not only the parameter of the models but also this statistics for the batch normalization layers

    return quantizedModel.state_dict(), pointsPerTensor, informationDict

def forward_and_backward(model, batch, idx_batch, epoch, criterion=None,
                         use_distillation_loss=False, teacher_model=None,
                         temperature_distillation=2, ask_teacher_strategy='always',
                         return_more_info=False):
    """
    batch: batch of data
    """

    #TODO: return_more_info is just there for backward compatibility. A big refactoring is due here, and there one should
    #remove the return_more_info flag

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        criterion = criterion.cuda()

    if use_distillation_loss is True and teacher_model is None:
        raise ValueError('To compute distillation loss you need to pass the teacher model')

    if not isinstance(ask_teacher_strategy, tuple):
        ask_teacher_strategy = (ask_teacher_strategy, )

    inputs, labels = batch
    labels = labels[:, 0]
    inputs = inputs.transpose(2, 1)
    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)

    if USE_CUDA:
        inputs = inputs.cuda()
        labels = labels.cuda()
        model = model.cuda()

    # forward + backward + optimize
    # outputs = model(inputs)
    outputs, trans, trans_feat = model(inputs)

    count_asked_teacher = 0

    if use_distillation_loss:
        #if cutoff_entropy_value_distillation is not None, we use the distillation loss only on the examples
        #whose entropy is higher than the cutoff.

        weight_teacher_loss = 0.7

        if 'entropy' in ask_teacher_strategy[0].lower():
            prob_out = torch.nn.functional.softmax(outputs).data
            entropy = [mhf.get_entropy(prob_out[idx_b, :]) for idx_b in range(prob_out.size(0))]

        if ask_teacher_strategy[0].lower() == 'always':
            mask_distillation_loss = torch.ByteTensor([True]*outputs.size(0))
        elif ask_teacher_strategy[0].lower() == 'cutoff_entropy':
            cutoff_entropy_value_distillation = ask_teacher_strategy[1]
            mask_distillation_loss = torch.ByteTensor([entr > cutoff_entropy_value_distillation for entr in entropy])
        elif ask_teacher_strategy[0].lower() == 'random_entropy':
            max_entropy = math.log2(outputs.size(1)) #max possible entropy that happens with uniform distribution
            mask_distillation_loss = torch.ByteTensor([random.random() < entr/max_entropy for entr in entropy])
        elif ask_teacher_strategy[0].lower() == 'incorrect_labels':
            _, predictions = outputs.max(dim=1)
            mask_distillation_loss = (predictions != labels).data.cpu()
        else:
            raise ValueError('ask_teacher_strategy is incorrectly formatted')

        #print(mask_distillation_loss.view(-1))
        #print(torch.arange(0, outputs.size(0)).size())

        #print(outputs.size())
        #index_distillation_loss = torch.arange(0, outputs.size(0))[mask_distillation_loss.view(-1, 1)].long()
        #inverse_idx_distill_loss = torch.arange(0, outputs.size(0))[1-mask_distillation_loss.view(-1, 1)].long()
        index_distillation_loss = torch.arange(0, outputs.size(0))[mask_distillation_loss.view(-1)].long()
        inverse_idx_distill_loss = torch.arange(0, outputs.size(0))[1-mask_distillation_loss.view(-1)].long()
        if USE_CUDA:
            index_distillation_loss = index_distillation_loss.cuda()
            inverse_idx_distill_loss = inverse_idx_distill_loss.cuda()

        # this criterion is the distillation criterion according to Hinton's paper:
        # "Distilling the Knowledge in a Neural Network", Hinton et al.

        softmaxFunction, logSoftmaxFunction, KLDivLossFunction  = nn.Softmax(dim=1), nn.LogSoftmax(dim=1), nn.KLDivLoss()
        if USE_CUDA:
            softmaxFunction, logSoftmaxFunction = softmaxFunction.cuda(), logSoftmaxFunction.cuda(),
            KLDivLossFunction = KLDivLossFunction.cuda()

        if index_distillation_loss.size() != torch.Size():
            count_asked_teacher = index_distillation_loss.numel()
            # if index_distillation_loss is not empty
            volatile_inputs = Variable(inputs.data[index_distillation_loss, :], requires_grad=False)
            if USE_CUDA: volatile_inputs = volatile_inputs.cuda()
            outputs_, _, _ = teacher_model(volatile_inputs)
            outputsTeacher = outputs_.detach()
            loss_masked = weight_teacher_loss * temperature_distillation**2 * KLDivLossFunction(
                    logSoftmaxFunction(outputs[index_distillation_loss, :]/ temperature_distillation),
                    softmaxFunction(outputsTeacher / temperature_distillation))
            loss_masked += (1-weight_teacher_loss) * criterion(outputs[index_distillation_loss, :],
                                                               labels[index_distillation_loss])
        else:
            loss_masked = 0

        if inverse_idx_distill_loss.size() != torch.Size([0]):
            #if inverse_idx_distill is not empty
            loss_normal = criterion(outputs[inverse_idx_distill_loss, :], labels[inverse_idx_distill_loss])
        else:
            loss_normal = 0

        loss = loss_masked + loss_normal

    else:
        loss = criterion(outputs, labels)

    loss.backward()

    if return_more_info:
        count_total = inputs.size(0)
        return loss.data, count_asked_teacher, count_total
    else:
        return loss.data

def evaluateModel(model, testLoader, fastEvaluation=True, maxExampleFastEvaluation=10000, k=1):

    'if fastEvaluation is True, it will only check a subset of *maxExampleFastEvaluation* images of the test set'

    if USE_CUDA:
        model = model.cuda()
    model.eval()
    correctClass = 0
    totalNumExamples = 0

    for idx_minibatch, data in enumerate(testLoader):

        # get the inputs
        inputs, labels = data
        inputs = inputs.transpose(2,1)
        labels = labels[:, 0]
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs, trans, trans_feat = model(inputs)

        _, topk_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
        topk_predictions = topk_predictions.t()
        correct = topk_predictions.eq(labels.view(1, -1).expand_as(topk_predictions))
        correctClass += correct.view(-1).float().sum(0, keepdim=True).data[0]
        totalNumExamples += len(labels)

        if fastEvaluation is True and totalNumExamples > maxExampleFastEvaluation:
            break

    return correctClass / totalNumExamples




if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
