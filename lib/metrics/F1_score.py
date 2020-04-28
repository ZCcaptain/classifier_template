from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
from overrides import overrides

classes = ['no', 'stick', 'gun', 'knife', 'dao']
class F1_abc(object):
    def __init__(self, cls_):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10
        self.cls = cls_

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B +
                                 self.C), self.A / self.B, self.A / self.C
        result = {"cls":self.cls, "precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(self, gold_labels,predictions):
        raise NotImplementedError



class F1_resnet(F1_abc):

    @overrides
    def __call__(self, gold_labels, predictions):

        tp = sum(pic_g == pic_p  and pic_g == self.cls for pic_g, pic_p in zip(gold_labels, predictions))
        res_g = sum( pic_g == self.cls for pic_g in gold_labels)
        res_p = sum( pic_p == self.cls for pic_p in predictions)

        self.A += tp
        self.B += res_p
        self.C += res_g
