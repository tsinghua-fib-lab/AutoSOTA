from proxies.autograd_hacks import *
from proxies.p_utils import get_layer_metric_array
from torch import nn
import copy

from common.constant import Config


class KNASEvaluator:

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:
        """
        https://github.com/Jingjing-NLP/KNAS/tree/main
        """
        import copy

        batch_size = 4

        grads = {}

        if space_name != Config.MLPSP:
            raise NotImplementedError
        assert isinstance(batch_data, dict)
        total_batches = batch_data["id"].shape[0] // batch_size
        for i in range(total_batches):
            # Sample the batch
            start_index = i * batch_size
            end_index = start_index + batch_size
            id_batch = batch_data["id"][start_index:end_index].to(device)
            value_batch = batch_data["value"][start_index:end_index].to(device)
            y_batch = batch_data["y"][start_index:end_index].type(torch.LongTensor).to(device)
            one_batch = {'id': id_batch, 'value': value_batch}

            loss_fn = nn.CrossEntropyLoss(reduction='mean').to(device)
            outputs = arch(one_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()

            # if not grad: return 0, 0, 0,batch_time.sum
            index_grad = 0
            index_name = 0
            for name, param in arch.named_parameters():
                if param.grad == None:
                    continue
                if index_name > 10: break
                if len(param.grad.view(-1).data[0:100]) < 50: continue
                index_grad = name
                index_name += 1
                # if index_name > 10: break
                # index_grad +=
                if name in grads:
                    grads[name].append(copy.copy(param.grad.view(-1).data[0:100]))
                else:
                    grads[name] = [copy.copy(param.grad.view(-1).data[0:100])]
            # print(index_grad)
            if len(grads[index_grad]) == 50:
                conv = 0
                maxconv = 0
                minconv = 0
                lower_layer = 1
                top_layer = 1
                para = 0
                for name in grads:
                    for i in range(50):  # nt(self.grads[name][0].size()[0])):
                        grad1 = torch.tensor([grads[name][k][i] for k in range(
                            25)])  # torch.tensor(grads[name][j],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25)],dtype=torch.float)
                        grad2 = torch.tensor([grads[name][k][i] for k in range(25,
                                                                               50)])  # torch.tensor(grads[name][i],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25,50)],dtype=torch.float)
                        grad1 = grad1 - grad1.mean()
                        grad2 = grad2 - grad2.mean()
                        conv += torch.dot(grad1,
                                          grad2) / 2500  # torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))#i#/i1.0*self.grads[name][0].size()[0]
                        para += 1
                break
        # Sum over all parameter's results to get the final score.
        return conv

    # def procedure(xloader, network, criterion, scheduler, optimizer, mode, grad=False):
    #     losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    #     if mode == 'train':
    #         network.train()
    #     elif mode == 'valid':
    #         network.eval()
    #     else:
    #         raise ValueError("The mode is not right : {:}".format(mode))
    #     grads = {}
    #
    #     data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
    #     for i, (inputs, targets) in enumerate(xloader):
    #         # if  > 50: break
    #         # if mode != 'train': break
    #         if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))
    #         if mode != 'train': return 0, 0, 0, time.time() - time.time()
    #         targets = targets.cuda(non_blocking=True)
    #         if mode == 'train': optimizer.zero_grad()
    #         # forward
    #         features, logits = network(inputs)
    #         loss = criterion(logits, targets)
    #         # backward
    #         # print(int(targets[0].data))
    #         # if int(targets[0].data) != 1: continue
    #         if mode == 'train':
    #             loss.backward()
    #             import copy
    #             # if not grad: return 0, 0, 0,batch_time.sum
    #             index_grad = 0
    #             index_name = 0
    #             for name, param in network.named_parameters():
    #                 # print(param.grad.view(-1).data)
    #                 if param.grad == None:
    #                     # print(name)
    #                     continue
    #                 # if param.grad.view(-1)[0] == 0 and param.grad.view(-1)[1] == 0: continue #print(name)
    #                 # print(i)
    #                 if index_name > 10: break
    #                 if len(param.grad.view(-1).data[0:100]) < 50: continue
    #                 index_grad = name
    #                 index_name += 1
    #                 # if index_name > 10: break
    #                 # index_grad +=
    #                 if name in grads:
    #                     grads[name].append(copy.copy(param.grad.view(-1).data[0:100]))
    #                 else:
    #                     grads[name] = [copy.copy(param.grad.view(-1).data[0:100])]
    #             # print(index_grad)
    #             if len(grads[index_grad]) == 50:
    #                 conv = 0
    #                 maxconv = 0
    #                 minconv = 0
    #                 lower_layer = 1
    #                 top_layer = 1
    #                 para = 0
    #
    #                 for name in grads:
    #                     # print(name)
    #                     '''for i in range(50):
    #                         grads[name][i] = torch.tensor(grads[name][i], dtype=torch.float)
    #                         #grads[name][i] = grads[name][i] - grads[name][i].mean()
    #                         #means += grads[name][i]
    #                     means = grads[name][0]
    #
    #                     for i in range(1,50):
    #                         means += grads[name][i]
    #                     conv = torch.abs(torch.dot(means, means)/2500)'''
    #                     for i in range(50):  # nt(self.grads[name][0].size()[0])):
    #                         # if len(grads[name])!=: print(name)
    #                         # for j in range(50):
    #                         # if i == j: continue
    #                         grad1 = torch.tensor([grads[name][k][i] for k in range(
    #                             25)])  # torch.tensor(grads[name][j],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25)],dtype=torch.float)
    #                         grad2 = torch.tensor([grads[name][k][i] for k in range(25,
    #                                                                                50)])  # torch.tensor(grads[name][i],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25,50)],dtype=torch.float)
    #                         grad1 = grad1 - grad1.mean()
    #                         grad2 = grad2 - grad2.mean()
    #                         conv += torch.dot(grad1,
    #                                           grad2) / 2500  # torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))#i#/i1.0*self.grads[name][0].size()[0]
    #                         para += 1
    #                 # conv /= para
    #                 # print("dot product: ", conv)
    #                 # print(conv, maxconv, minconv)# top_layer/lower_layer)
    #                 # print("endddddddddd")
    #                 # count += 1
    #                 break
    #             # else: print(name)
    #             # optimizer.step()
    #             # optimizer.step()
    #         ## record loss and accuracy
    #         # prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    #         # losses.update(loss.item(),  inputs.size(0))
    #         # top1.update  (prec1.item(), inputs.size(0))
    #         # top5.update  (prec5.item(), inputs.size(0))
    #         # count time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #     if mode == 'train':
    #         return conv, 0, 0, batch_time.sum  # conv, maxconv, minconv
    #     else:
    #         return 0, 0, 0, batch_time.sum
