import torch
from torch.optim.optimizer import Optimizer, required
import copy
import math

class FTP(object):
    def __init__(self, k=1.0, exclude_set={}):
        self.exclude_set = exclude_set
        self.threshold = torch.nn.Hardtanh(0,1)
        self.k = k # Gradient annealing factor
        self.j = 0 # Buffer counter

        # AdamUtil parameteres
        self.mu = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 1
        
        # Buffers
        self.gamma_buff = []
        self.first_m_gamma = []
        self.second_m_gamma = []
        self.prev_c = []
        self.prev_scale = []
    
    @torch.no_grad()
    def step(self,name, curr, pre, d_p):
        if curr.requires_grad and name not in self.exclude_set:
            c_t = (curr - d_p) - pre
            norms = self._mars_norm(c_t)

            if self.t == 1:
                gamma = torch.tensor(1e-8).to(norms.device)
                self._update_buffers(gamma)
            else:
                # Get previous values
                gamma_prev = self.gamma_buff[self.j]
                c_prev = self.prev_c[self.j]
                scale_prev = self.prev_scale[self.j]

                # Calculate gradient for gamma
                gamma_grad = torch.sum(self._dot(curr.grad, c_prev, scale = scale_prev))

                # Anneal positive gradient 
                if gamma_grad > 0:
                    gamma_grad = gamma_grad * self.k

                gamma = self._adam_util(gamma_prev, gamma_grad)

                # Clip gamma
                gamma = self._clip(gamma, norms)
            
            # Update
            denom = 1/norms
            ratio = gamma * denom
            new_p = pre + self.threshold(ratio) * c_t 

            # Save updated values
            self._update_buffers(gamma, c_t, denom)
            self.j += 1
            return new_p
        else:
            return None
        
    def incre_counters(self):
        self.t += 1
        self.j = 0

    @torch.no_grad()
    def _mars_norm(self,tensor):
        return torch.sum(torch.abs(tensor), dim=tuple(range(1,tensor.dim())), keepdim=True) + 1e-8
    
    @torch.no_grad()
    def _clip(self, constraint, norms):
        return torch.nn.functional.hardtanh(constraint,1e-8,norms.max())
        
    @torch.no_grad()
    def _dot(self,tensor1,tensor2,scale=1):
        return torch.sum(torch.mul(tensor1, tensor2),dim=tuple(range(1,tensor1.dim())),keepdim=True)*scale
    
    @torch.no_grad()
    def _adam_util(self,prev, grad):
        first_moment = self.beta1 * self.first_m_gamma[self.j] + (1-self.beta1) * grad
        second_moment = self.beta2 * self.second_m_gamma[self.j] + (1-self.beta2) * grad**2
        self.first_m_gamma[self.j] = first_moment
        self.second_m_gamma[self.j] = second_moment
        first_moment = first_moment/(1-self.beta1**self.t)
        second_moment = second_moment/(1-self.beta2**self.t)
        return prev  - self.mu * first_moment/(torch.sqrt(second_moment)+1e-8)
    
    def _update_buffers(self,gamma,c_t=None,denom=None):
        if c_t is None:
            self.first_m_gamma.append(0.0)
            self.second_m_gamma.append(0.0)
            self.gamma_buff.append(gamma)
            self.prev_c.append(0.0)
            self.prev_scale.append(0.0)
        else:
            self.gamma_buff[self.j] = gamma
            self.prev_c[self.j] = c_t
            self.prev_scale[self.j] = denom

            
class SGDP(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, k=1.0, exclude_set = {}):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDP, self).__init__(params, defaults)
        self.first_iter_flag = False
         
        # initialize FTP
        self.ftp = FTP(k, exclude_set=exclude_set)
                    

    def __setstate__(self, state):
        super(SGDP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p, name, pre in zip(group['params'],group['name'],group['pre']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening) 
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # FTP step
                d_p = group['lr']*d_p
                new_p = self.ftp.step(name,p,pre,d_p)
                if new_p is not None :
                    p.copy_(new_p)
                else:
                    p.add_(d_p, alpha=-1)
        # FTP increment internal counters
        self.ftp.incre_counters()        
        return loss


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, k=1.0, exclude_set={}):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamP, self).__init__(params, defaults)

        # initialize FTP
        self.ftp = FTP(k, exclude_set=exclude_set)

    def __setstate__(self, state):
        super(AdamP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(group,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        # FTP increment internal counters            
        self.ftp.incre_counters()    
        return loss

    def adam(self, group,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps, 
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
        
        i = 0
        for param, name, pre in zip(group['params'],group['name'],group['pre']):
            if param.grad is None: 
                continue
            grad = param.grad
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            i += 1
            
            # FTP step
            d_p = step_size * exp_avg/denom + lr * weight_decay * param
            new_p = self.ftp.step(name,param,pre,d_p)
            if new_p is None :
                new_p = param - d_p
            param.copy_(new_p)
