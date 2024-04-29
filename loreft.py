import torch
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from pyvene.models.layers import LowRankRotateLayer
from transformers.activations import ACT2FN


class LoreftIntervention(SourcelessIntervention, TrainableIntervention,
                         DistributedRepresentationIntervention):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)

        # Init R (rxd rotation matrix)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])

        # Make R orthogonal
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

        # Init linear transformation W_ + b ==> learned projected source
        # Creates linear layer mapping embed_dim (d) to low-rank dim (r)
        self.Wb = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        # Init dropout layer for regularization
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

        # Get activation function for intervention layer
        # If not specified, use linear activation
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
    def forward(
        self, base, source=None, subspaces=None
    ):
        # Rotated base
        Rh = self.rotate_layer(base)
        # Compute h + R^T(Wh + b − Rh)
        output = base + torch.matmul(
            (self.act_fn(self.Wb(base)) - Rh), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return