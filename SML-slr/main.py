#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path


savedir = Path('experiments') / Path(str(int(time.time())))
if not os.path.exists(savedir):
            os.makedirs(savedir)

history = {
    'train_loss' : [],
    'test_loss' : [],
    'test_acc' : []
}
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#     def forward(self, x, target, smoothing=0.1):
#         confidence = 1. - smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + smoothing * smooth_loss
#         return loss.mean()

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='config/train_joint_joint_motion.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--train-feeder-args1',
        default=dict(),
        help='the arguments of data loader for training')
    # parser.add_argument(
    #     '--train-feeder-args2',
    #     default=dict(),
    #     help='the arguments of data loader for training')
    # parser.add_argument(
    #     '--train-feeder-args3',
    #     default=dict(),
    #     help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--test-feeder-args1',
        default=dict(),
        help='the arguments of data loader for test')
    # parser.add_argument(
    #     '--test-feeder-args2',
    #     default=dict(),
    #     help='the arguments of data loader for test')
    # parser.add_argument(
    #     '--test-feeder-args3',
    #     default=dict(),
    #     help='the arguments of data loader for test')
    
    # graph
    parser.add_argument('--graph', default=None)
    parser.add_argument('--graph_args', default=None)

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--keep_rate',
        type=float,
        default=0.9,
        help='keep probability for drop')
    parser.add_argument(
        '--groups',
        type=int,
        default=8,
        help='decouple groups')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.pre_predictions = None
        arg.model_saved_name = "save_models/" + arg.Experiment_name
        arg.work_dir = "./work_dir/" + arg.Experiment_name
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input(
                            'Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    # def load_data(self):
    #     Feeder = import_class(self.arg.feeder)
    #     self.data_loader = dict()
    #     if self.arg.phase == 'train':
    #         self.data_loader['train'] = torch.utils.data.DataLoader(
    #             dataset=Feeder(**self.arg.train_feeder_args),
    #             batch_size=self.arg.batch_size,
    #             shuffle=True,
    #             num_workers=self.arg.num_worker,
    #             drop_last=True,
    #             worker_init_fn=init_seed)
    #     self.data_loader['test'] = torch.utils.data.DataLoader(
    #         dataset=Feeder(**self.arg.test_feeder_args),
    #         batch_size=self.arg.test_batch_size,
    #         shuffle=False,
    #         num_workers=self.arg.num_worker,
    #         drop_last=False,
    #         worker_init_fn=init_seed)


#
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        feed_dataset = Feeder(**self.arg.train_feeder_args)
        feed_dataset.append_dataset(**self.arg.train_feeder_args1)
        # feed_dataset.append_dataset(**self.arg.train_feeder_args2)
        # feed_dataset.append_dataset(**self.arg.train_feeder_args3)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset = feed_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        feed_dataset_test = Feeder(**self.arg.test_feeder_args)
        feed_dataset_test.append_dataset(**self.arg.test_feeder_args1)
        # feed_dataset_test.append_dataset(**self.arg.test_feeder_args2)
        # feed_dataset_test.append_dataset(**self.arg.test_feeder_args3)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=feed_dataset_test,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        # output_device = 'cpu'
        self.output_device = output_device
        Graph = import_class(self.arg.graph)
        self.graph = Graph(**self.arg.graph_args)
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(graph = self.graph, **self.arg.model_args).cuda(output_device)
        self.dis_model = Model(graph = self.graph, **self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.dis_loss = nn.KLDivLoss().cuda(output_device)
        # self.loss = LabelSmoothingCrossEntropy().cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir + '/eval_results')

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.temperature =10
        self.flag = False
        if os.path.exists(self.arg.model_saved_name + '-' + 'best' + '.pt'):
            self.weight_path = self.arg.model_saved_name + '-' + 'best' + '.pt'
            self.dis_model.load_state_dict(torch.load(self.weight_path))
            self.flag = True
            print('best_pt exist!')

        train_loss = 0
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if epoch >= self.arg.only_train_epoch:
            print('only train part, require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            print('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            for i in range(len(data)):
                data[i] = Variable(data[i].float().cuda(
                    self.output_device), requires_grad=False)
            # data = Variable(data.float().cuda(
            #     self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(
                self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            if epoch < 100:
                keep_prob = -(1 - self.arg.keep_rate) / 100 * epoch + 1.0
            else:
                keep_prob = self.arg.keep_rate
            # output,jm,bm,j,b,jmotion,bmotion = self.model(data[0] ,keep_prob)
            output = self.model(data[0],data[1])
            if self.flag:
                self.dis_model.eval()
                with torch.no_grad():
                    self.pre_predictions = self.dis_model(data[0],data[1])

            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            if self.pre_predictions != None:
                #print('using self studying')
                loss = 0.5*self.loss(output, label)+ 0.5*self.dis_loss(F.softmax(self.pre_predictions / self.temperature,dim=1),F.softmax(output / self.temperature, dim=1)) + l1
                # loss = self.loss(output, label)
            else:
                loss = self.loss(output, label) + l1
            train_loss += loss.detach().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data)
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())

            self.lr = self.optimizer.param_groups[0]['lr']
            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        history['train_loss'].append(train_loss)

        # state_dict = self.model.state_dict()
        # weights = OrderedDict([[k.split('module.')[-1],
        #                         v.cpu()] for k, v in state_dict.items()])
        #
        # torch.save(weights, self.arg.model_saved_name +
        #            '-' + str(epoch) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            self.print_log('Eval epoch: {}'.format(epoch + 1))
            for ln in loader_name:
                loss_value = []
                score_frag = []
                right_num_total = 0
                total_num = 0
                loss_total = 0
                step = 0
                process = tqdm(self.data_loader[ln])

                for batch_idx, (data, label, index) in enumerate(process):
                    for i in range(len(data)):
                        data[i] = Variable(data[i].float().cuda(
                            self.output_device), requires_grad=False)
                    # data = Variable(
                    #     data.float().cuda(self.output_device),
                    #     requires_grad=False)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False)

                    with torch.no_grad():
                        output = self.model(data[0],data[1])

                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    test_loss += loss.item()
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' +
                                        str(x) + ',' + str(true[i]) + '\n')
                score = np.concatenate(score_frag)

                if 'UCLA' in arg.Experiment_name:
                    self.data_loader[ln].dataset.sample_name = np.arange(
                        len(score))
                history['test_loss'].append(test_loss)
                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                history['test_acc'].append(accuracy)
                if accuracy > self.best_acc:

                    self.best_acc = accuracy
                    score_dict = dict(
                        zip(self.data_loader[ln].dataset.sample_name, score))

                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1],
                                            v.cpu()] for k, v in state_dict.items()])

                    if not os.path.exists(self.arg.model_saved_name +
                               '-' + 'best' + '.pt'):
                        os.system(r"touch{}".format(self.arg.model_saved_name +
                               '-' + 'best' + '.pt'))

                    torch.save(weights, self.arg.model_saved_name +
                               '-' + 'best' + '.pt')

                    with open('./work_dir/' + arg.Experiment_name + '/eval_results/best_acc' + '.pkl'.format(
                            epoch, accuracy), 'wb') as f:
                        pickle.dump(score_dict, f)



                print('Eval Accuracy: ', accuracy,
                    ' model: ', self.arg.model_saved_name)

                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

                # with open('./work_dir/' + arg.Experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                #         epoch, accuracy), 'wb') as f:
                #     pickle.dump(score_dict, f)
        return np.mean(loss_value)
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = int(self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size)
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                val_loss = self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

                # self.lr_scheduler.step(val_loss)
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
            ax1.plot(history['train_loss'])
            ax1.plot(history['test_loss'])
            ax1.legend(['Train', 'Test'], loc='upper left')
            ax1.set_xlabel('Epoch')
            ax1.set_title(arg.model_saved_name + 'Loss')

            ax2.set_title('Model accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.plot(history['test_acc'])
            xmax = np.argmax(history['test_acc'])
            ymax = np.max(history['test_acc'])
            text = "x={}, y={:.3f}".format(xmax, ymax)
            ax2.annotate(text, xy=(xmax, ymax))

            fig.tight_layout()
            fig.savefig(str(savedir / "perf.png"))

            print('best accuracy: ', self.best_acc,
                  ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=self.arg.start_epoch, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                # assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
