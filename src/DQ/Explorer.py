import math

class ExponentialExplorer(object):
    def __init__(self, eps_end, eps_start, eps_decay, start_reduction):
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.start_reduction = start_reduction

    def __call__(self, step):
        if step > self.start_reduction:
            eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * (step-self.start_reduction) / self.eps_decay)
            if step % 1000 == 0:
                print(eps)
            return eps
        else:
            return self.eps_start

class LinearExplorer(object):
    def __init__(self, eps_end, eps_start, eps_decay, start_reduction):
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.start_reduction = start_reduction

    def __call__(self, step):
        if step > self.start_reduction:
            eps = self.eps_end + (self.eps_start - self.eps_end) *(max([(self.eps_decay-step+self.start_reduction)/self.eps_decay,0]))
            if step % 1000 == 0:
                print(eps)
            return eps
        else:
            return self.eps_start