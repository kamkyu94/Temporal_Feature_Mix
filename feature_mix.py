import torch
import torch.nn as nn
import yolox.models.network_blocks as nb


class FeatureMix(nn.Module):
    def __init__(self, model, batch_size, prob, ratio):
        super().__init__()
        # Set
        self.model = model
        self.batch_size = batch_size
        self.prob = prob
        self.ratio = ratio

        # Initialize 1
        self.mix = False
        self.record = False
        self.outputs = {}

        # Initialize 2
        self.module_list = self.get_children(model, nb.TFM)
        self.module_num = len(self.module_list)
        self.sample_module_num = int(round(self.module_num * self.prob))
        self.mix_table = None

        # Add hooks
        self.forward_hooks = []
        for i, m in enumerate(self.module_list):
            self.forward_hooks.append(m.register_forward_hook(self.save_outputs_hook(i)))

    def get_children(self, model, target_module):
        # Check
        children = list(model.children())
        flattened_children = []

        # If model is the last child
        if len(children) == 0:
            if type(model) == target_module:
                return model
            else:
                return []

        # Look for children with recursion
        else:
            for child in children:
                try:
                    flattened_children.extend(self.get_children(child, target_module))
                except TypeError:
                    flattened_children.append(self.get_children(child, target_module))

        return flattened_children

    def save_outputs_hook(self, module_idx):
        def hook_fn(module, inputs, outputs):
            # Mix mode
            if self.mix and not self.record and module_idx in self.outputs.keys():
                # Get saved features
                saved_feature = self.outputs[module_idx]

                # Temporal Feature Mix
                a = self.mix_table[:, module_idx:module_idx+1] * torch.rand((self.batch_size, outputs.shape[1]))
                a = a * self.ratio
                a = a.view(self.batch_size, outputs.shape[1], 1, 1).to(outputs.device).half()
                outputs = (1 - a) * outputs + a * saved_feature

                return outputs

            # Record mode
            elif self.record and not self.mix:
                self.outputs[module_idx] = outputs.clone().detach()
                return None

            return outputs

        return hook_fn

    def remove_hooks(self):
        for fh in self.forward_hooks:
            fh.remove()
        del self.outputs

    def start_feature_record(self):
        self.mix = False
        self.record = True

    def end_feature_record(self):
        self.mix = False
        self.record = False

    def start_feature_mix(self):
        self.mix = True
        self.record = False

        # Initialize mix table
        self.mix_table = torch.zeros((self.batch_size, self.module_num)).detach().float()

        # Set mix table
        for bdx in range(self.batch_size):
            layer_idx = torch.randperm(self.module_num)[:self.sample_module_num]
            self.mix_table[bdx, layer_idx] = 1.

    def end_feature_mix(self):
        self.mix = False
        self.record = False
        self.mix_table = None
        self.outputs = {}

    def forward(self, x, targets):
        return self.model(x, targets)
