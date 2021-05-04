# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PathLayer(nn.Module):

    def __init__(self, unit_count, task_count, unit_mapping, name="pathlayer"):

        super(PathLayer, self).__init__()

        self.use_path = True
        self.name = name
        self.unit_count = unit_count
        unit_mapping = unit_mapping
        self.active_task = 0
        self._unit_mapping = torch.nn.Parameter(unit_mapping)

    def get_unit_mapping(self):

        return self._unit_mapping

    def set_active_task(self, active_task):

        self.active_task = active_task
        return active_task

    def forward(self, input):

        if not self.use_path:
            return input

        mask = torch.index_select(self._unit_mapping, 0, (torch.ones(input.shape[0])*self.active_task).long().to(device))\
            .unsqueeze(2).unsqueeze(3)
        input.data.mul_(mask)

        return input
