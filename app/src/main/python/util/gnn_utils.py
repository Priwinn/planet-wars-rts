import torch
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.utils import to_dense_batch, add_self_loops

def owner_one_hot_encoding(owner: torch.Tensor, player_id: int) -> torch.Tensor:
    """Convert owner integer to one-hot encoding. Assume Neutral=0, Controlled=1, Opponent=2 (swaps controlled and opponent if needed)"""
    one_hot = torch.nn.functional.one_hot(
        owner.long(), num_classes=3
    )
    # Swap controlled and opponent if needed
    if player_id == 2:
        one_hot = one_hot[..., [0, 2, 1]]
    return one_hot

def preprocess_graph_data(graph_data: list[PyGData], 
                          player_id: int,
                          use_tick: bool,
                          return_mask: bool = True):
    input = PyGBatch.from_data_list(graph_data)
    planet_owners = input.x[:, 0]
    transporter_owners_per_edge = input.edge_attr[:, 0]
    transporter_owners = input.x[:, 3]
    if return_mask:
        source_mask = to_dense_batch(torch.logical_and(planet_owners == 1, transporter_owners == 0), input.batch, fill_value=False)[0]
        source_mask = torch.cat((torch.ones(input.batch_size, 1, dtype=torch.bool, device=source_mask.device), source_mask), dim=1)
    if use_tick:
        input.x = torch.cat((owner_one_hot_encoding(planet_owners, player_id),
                        input.x[:, 1:-1],
                        input.tick[input.batch].unsqueeze(-1)),
                        dim=-1)
    else:
        input.x = torch.cat((owner_one_hot_encoding(planet_owners, player_id),
                        input.x[:, 1:-1]), dim=-1)
    input.edge_attr = torch.cat((owner_one_hot_encoding(transporter_owners_per_edge, player_id),
                            input.edge_attr[:, 1:]), dim=-1)
    input.edge_index, input.edge_attr = add_self_loops(input.edge_index, input.edge_attr, fill_value='mean')
    if return_mask:
        return input, source_mask
    else:
        return input