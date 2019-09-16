import datasets
import os
import copy

def __mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

datasets.BASE_DATA_FOLDER = 'data'

batch_size = 50
epochsToTrainCIFAR = 100
TRAIN_TEACHER_MODEL = False
TRAIN_DISTILLED_MODEL = True
TRAIN_SMALLER_MODEL = True
TRAIN_DISTILLED_QUANTIZED_MODEL = True

cifar10 = datasets.CIFAR10() #-> will be saved in /home/saved_datasets/cifar10
train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)

import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec, useBatchNorm=True, useAffineTransformInBatchNorm=True)
#convForwModel.train_model(teacherModel, train_loader, test_loader, epochs_to_train=20)

import cnn_models.conv_forward_model as convForwModel
import cnn_models.help_fun as cnn_hf
import model_manager
model_manager_path = 'model_manager_cifar10.tst'
model_save_path ='models'
__mkdir(model_save_path)

if os.path.exists(model_manager_path):
    cifar10Manager = model_manager.ModelManager('model_manager_cifar10.tst',
                                            'model_manager', create_new_model_manager=False)#the first t
else:
    cifar10Manager = model_manager.ModelManager('model_manager_cifar10.tst',
                                            'model_manager', create_new_model_manager=True)#the first t


model_name = 'cifar10_teacher'
teacherModelPath = os.path.join(model_save_path, model_name)
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=True,
                                              useAffineTransformInBatchNorm=True)
print(cifar10Manager.saved_models)

#convForwModel.train_model(teacherModel, train_loader, test_loader, epochs_to_train=2)
if not model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(model_name, teacherModelPath,
            arguments_creator_function={**convForwModel.teacherModelSpec,
                                        'useBatchNorm':True,
                                        'useAffineTransformInBatchNorm':True})

model_path  = cifar10Manager.saved_models[model_name][0][0]
print(model_path)
if TRAIN_TEACHER_MODEL:

    cifar10Manager.train_model(teacherModel, model_name=model_name,
                            train_function=convForwModel.train_model,
                            arguments_train_function={'epochs_to_train': epochsToTrainCIFAR},
                            train_loader=train_loader, test_loader=test_loader)
    # else:
    #     cifar10Manager.train_model(teacherModel, model_name=model_name,
    #                         train_function=convForwModel.train_model,
    #                         continue_training_from =1,
    #                         arguments_train_function={'epochs_to_train': epochsToTrainCIFAR},
    #                         train_loader=train_loader, test_loader=test_loader)

    print("Teacher Model Training complete")

print("Eval Teacher model")
print(model_name)
teacherModel.load_state_dict(cifar10Manager.load_model_state_dict(model_name))
acc = cnn_hf.evaluateModel(teacherModel, test_loader, k=1)
print("Top-1 eval acc is {}".format(acc))

smallerModelSpec2 = {'spec_conv_layers': [(25, 5, 5), (10, 5, 5), (10, 5, 5), (5, 5, 5)],
                    'spec_max_pooling': [(1, 2, 2), (3, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (4, 0.4)],
                    'spec_linear': [300], 'width': 32, 'height': 32}

small_model_name = 'cifar10_smaller_spec2'
model_small_spec = copy.deepcopy(smallerModelSpec2)

smallerModelPath = os.path.join(model_save_path, small_model_name)
smallerModel = convForwModel.ConvolForwardNet(**model_small_spec,
                                                useBatchNorm=True,
                                                useAffineTransformInBatchNorm=True)
if not small_model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(small_model_name, smallerModelPath,
            arguments_creator_function={**model_small_spec,
                                        'useBatchNorm':True,
                                        'useAffineTransformInBatchNorm':True})


if TRAIN_SMALLER_MODEL:

    cifar10Manager.train_model(smallerModel, model_name=small_model_name,
                                    train_function=convForwModel.train_model,
                                    arguments_train_function={'epochs_to_train': epochsToTrainCIFAR},
                                    train_loader=train_loader, test_loader=test_loader)
    print("SMALLER Model Training complete")
    smallerModel.load_state_dict(cifar10Manager.load_model_state_dict(small_model_name))
    acc = cnn_hf.evaluateModel(smallerModel, test_loader, k=1)
    print("Top-1 eval acc of smaller model is {}".format(acc))


distilled_model_name = 'cifar10_distilled_spec2'
distilledModelSpec = copy.deepcopy(smallerModelSpec2)
distilledModelSpec['spec_dropout_rates'] = [] #no dropout with distilled model

distilledModelPath = os.path.join(model_save_path, distilled_model_name)
distilledModel = convForwModel.ConvolForwardNet(**distilledModelSpec,
                                                useBatchNorm=True,
                                                useAffineTransformInBatchNorm=True)

if not distilled_model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(distilled_model_name, distilledModelPath,
            arguments_creator_function={**distilledModelSpec,
                                        'useBatchNorm':True,
                                        'useAffineTransformInBatchNorm':True})


if TRAIN_DISTILLED_MODEL:

    cifar10Manager.train_model(distilledModel, model_name=distilled_model_name,
                                train_function=convForwModel.train_model,
                                arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                                                            'teacher_model':teacherModel,
                                                            'use_distillation_loss':True},
                                train_loader=train_loader, test_loader=test_loader)
    print("DISTILLED Model Training complete")

print("Eval DISTILLED model")

distilledModel.load_state_dict(cifar10Manager.load_model_state_dict(distilled_model_name))
acc = cnn_hf.evaluateModel(distilledModel, test_loader, k=1)
print("Top-1 eval acc is {}".format(acc))


if TRAIN_DISTILLED_QUANTIZED_MODEL :
    numBits = [8]
    for numBit in numBits:
        distilled_quantized_model_name = 'cifar10_distilled_spec2_quantized{}bits'.format(numBit)

        distilled_quantized_model_path = os.path.join(model_save_path, distilled_quantized_model_name)
        distilled_quantized_model = convForwModel.ConvolForwardNet(**distilledModelSpec,
                                                        useBatchNorm=True,
                                                        useAffineTransformInBatchNorm=True)
        if not distilled_quantized_model_name in cifar10Manager.saved_models:
            cifar10Manager.add_new_model(distilled_quantized_model_name, distilled_quantized_model_path,
                                            arguments_creator_function={**distilledModelSpec,
                                                                        'useBatchNorm': True,
                                                                        'useAffineTransformInBatchNorm': True})

        cifar10Manager.train_model(distilled_quantized_model, model_name=distilled_quantized_model_name,
                                    train_function=convForwModel.train_model,
                                    arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                                                                'teacher_model': teacherModel,
                                                                'use_distillation_loss': True,
                                                                'quantizeWeights':True,
                                                                'numBits':numBit,
                                                                'bucket_size':256},
                                    train_loader=train_loader, test_loader=test_loader)
        distilled_quantized_model.load_state_dict(cifar10Manager.load_model_state_dict(distilled_quantized_model_name))
        acc = cnn_hf.evaluateModel(distilled_quantized_model, test_loader, k=1)
        print("Top-1 eval {} acc is {}".format(distilled_quantized_model_name, acc))

