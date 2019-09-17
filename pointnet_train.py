# import datasets
import model_manager
import cnn_models.conv_forward_model as convForwModel
import os
import copy
import argparse
import torch
import pointnet
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer


def __mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# datasets.BASE_DATA_FOLDER = 'data'


batch_size = 50
# epochsToTrainCIFAR = 100
epochsToTrain = 10
TRAIN_TEACHER_MODEL = False
TRAIN_DISTILLED_MODEL = False
TRAIN_SMALLER_MODEL = True
TRAIN_DISTILLED_QUANTIZED_MODEL = False

# cifar10 = datasets.CIFAR10() #->
# train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str,
                    default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform',
                    action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


# train_loader, test_loader = cifar10.getTrainLoader(batch_size), cifar10.getTestLoader(batch_size)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

model_manager_path = 'model_manager_pointnetcls.tst'
manager_name = "mg_pointnetcls"
model_save_path = 'models'
__mkdir(model_save_path)

if os.path.exists(model_manager_path):
    pointNetManager = model_manager.ModelManager(model_manager_path,
                                                 manager_name, create_new_model_manager=False)  # the first t
else:
    pointNetManager = model_manager.ModelManager(model_manager_path,
                                                 manager_name, create_new_model_manager=True)  # the first t

model_name = 'pointnetcls_teacher'
teacherModelPath = os.path.join(model_save_path, model_name)
classifier = PointNetCls(
    k=num_classes, feature_transform=opt.feature_transform)
teacherModel = classifier
print(pointNetManager.saved_models)

#convForwModel.train_model(teacherModel, train_loader, test_loader, epochs_to_train=2)
if not model_name in pointNetManager.saved_models:
    pointNetManager.add_new_model(model_name, teacherModelPath,
                                  arguments_creator_function={
                                      #  **convForwModel.teacherModelSpec,
                                      #  'useBatchNorm': True,
                                      #  'useAffineTransformInBatchNorm': True
                                  })

model_path = pointNetManager.saved_models[model_name][0][0]
print(model_path)
if TRAIN_TEACHER_MODEL:

    pointNetManager.train_model(teacherModel, model_name=model_name,
                                train_function=pointnet.model.train_model,
                                arguments_train_function={
                                    'epochs_to_train': epochsToTrain},
                                train_loader=train_loader, test_loader=test_loader)
    # else:
    #     pointNetManager.train_model(teacherModel, model_name=model_name,
    #                         train_function=convForwModel.train_model,
    #                         continue_training_from =1,
    #                         arguments_train_function={'epochs_to_train': epochsToTrain},
    #                         train_loader=train_loader, test_loader=test_loader)

    print("Teacher Model Training complete")

print("Eval Teacher model")
print(model_name)
teacherModel.load_state_dict(pointNetManager.load_model_state_dict(model_name))
acc = pointnet.model.evaluateModel(teacherModel, test_loader, k=1)
print("Top-1 eval acc is {}".format(acc))

small_model_name = "small_pointnet_same"
smallerModelPath = os.path.join(model_save_path, small_model_name)
smallerModel = PointNetCls(
    k=num_classes, feature_transform=opt.feature_transform)

# smallerModel = convForwModel.ConvolForwardNet(**model_small_spec,
#                                               useBatchNorm=True,
#                                               useAffineTransformInBatchNorm=True)
if not small_model_name in pointNetManager.saved_models:
    pointNetManager.add_new_model(small_model_name, smallerModelPath,
                                  arguments_creator_function={'k': num_classes,
                                                              'feature_transform': opt.feature_transform})
#                                                              'useAffineTransformInBatchNorm': True})

if TRAIN_SMALLER_MODEL:
    pointNetManager.train_model(smallerModel, model_name=small_model_name,
                                train_function=pointnet.model.train_model,
                                arguments_train_function={
                                    'epochs_to_train': epochsToTrain},
                                train_loader=train_loader, test_loader=test_loader)
    print("SMALLER Model Training complete")
    smallerModel.load_state_dict(
        pointNetManager.load_model_state_dict(small_model_name))
    acc = pointnet.model.evaluateModel(smallerModel, test_loader, k=1)
    print("Top-1 eval acc of smaller model is {}".format(acc))


distilled_model_name = 'distill_pointnet'
# distilledModelSpec = copy.deepcopy(smallerModelSpec2)
# no dropout with distilled model
# distilledModelSpec['spec_dropout_rates'] = []

distilledModelPath = os.path.join(model_save_path, distilled_model_name)
distilledModel = PointNetCls(
    k=num_classes, feature_transform=opt.feature_transform)

if not distilled_model_name in pointNetManager.saved_models:
    pointNetManager.add_new_model(distilled_model_name, distilledModelPath,
                                  arguments_creator_function={'k': num_classes, 'feature_transform': opt.feature_transform})

if TRAIN_DISTILLED_MODEL:

    pointNetManager.train_model(distilledModel, model_name=distilled_model_name,
                                train_function=pointnet.model.train_model,
                                arguments_train_function={'epochs_to_train': epochsToTrain,
                                                          'teacher_model': teacherModel,
                                                          'use_distillation_loss': True},
                                train_loader=train_loader, test_loader=test_loader)
    print("DISTILLED Model Training complete")

print("Eval DISTILLED model")

distilledModel.load_state_dict(
    pointNetManager.load_model_state_dict(distilled_model_name))
acc = pointnet.model.evaluateModel(distilledModel, test_loader, k=1)
print("Top-1 eval acc is {}".format(acc))


if TRAIN_DISTILLED_QUANTIZED_MODEL:
    numBits = [8]
    for numBit in numBits:
        distilled_quantized_model_name = 'cifar10_distilled_spec2_quantized{}bits'.format(
            numBit)

        distilled_quantized_model_path = os.path.join(
            model_save_path, distilled_quantized_model_name)
        distilled_quantized_model = pointnet.model.ConvolForwardNet(**distilledModelSpec,
                                                                    useBatchNorm=True,
                                                                    useAffineTransformInBatchNorm=True)
        if not distilled_quantized_model_name in pointNetManager.saved_models:
            pointNetManager.add_new_model(distilled_quantized_model_name, distilled_quantized_model_path,
                                          arguments_creator_function={**distilledModelSpec,
                                                                      'useBatchNorm': True,
                                                                      'useAffineTransformInBatchNorm': True})

        pointNetManager.train_model(distilled_quantized_model, model_name=distilled_quantized_model_name,
                                    train_function=pointnet.model.train_model,
                                    arguments_train_function={'epochs_to_train': epochsToTrain,
                                                              'teacher_model': teacherModel,
                                                              'use_distillation_loss': True,
                                                              'quantizeWeights': True,
                                                              'numBits': numBit,
                                                              'bucket_size': 256},
                                    train_loader=train_loader, test_loader=test_loader)
        distilled_quantized_model.load_state_dict(
            pointNetManager.load_model_state_dict(distilled_quantized_model_name))
        acc = pointnet.model.evaluateModel(
            distilled_quantized_model, test_loader, k=1)
        print("Top-1 eval {} acc is {}".format(distilled_quantized_model_name, acc))
