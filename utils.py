from model import ReftModel
"""
Shared functions between PIQA and FACTOID runs.
"""
def print_logistics(task, model, layers_config, rank, position, epochs,
                           max_length, allow_cls_grad):
    print(f"task: {task}, model: {model},"
          f"layers: {layers_config}, rank: {rank}, "
          f"position: {position}, epoch: {epochs}, "
          f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}")
