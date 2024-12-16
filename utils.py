import torch


def get_model_params(node):
    return torch.nn.utils.parameters_to_vector(node.model.parameters())


def get_diff_vectorized(m1, others):
    """
    Vectorized computation of normalized differences between one model's parameters
    and multiple other models' parameters.

    Args:
        m1 (torch.Tensor): Parameters of the reference model
        others (torch.Tensor): Stacked parameters of other models

    Returns:
        torch.Tensor: Vector of normalized differences
    """
    # Expand m1 to match others' shape for broadcasting
    m1_expanded = m1.unsqueeze(0)

    # Compute differences in one go
    differences = m1_expanded - others

    # Compute norms efficiently using torch.norm
    diff_norms = torch.norm(differences, dim=1)
    m1_norm = torch.norm(m1)

    # Return normalized differences
    return diff_norms / m1_norm


def convergence_loss(node_id, nodes, num_nodes, device="cpu"):
    """
    Compute the average parameter difference between one node and all other nodes.
    Optimized for parallel computation on GPU if available.

    Args:
        node_id (int): ID of the current node
        nodes (list): List of all nodes
        num_nodes (int): Total number of nodes
        device (str): Device to perform computations on ('cuda' or 'cpu')

    Returns:
        torch.Tensor: Average normalized difference between current node and others
    """
    # Get current node parameters
    curr_params = get_model_params(nodes[node_id])

    # Pre-allocate tensor for other nodes' parameters
    other_params = torch.empty(
        (num_nodes - 1, curr_params.size(0)), device=device, dtype=curr_params.dtype
    )

    # Fill the tensor with parameters from other nodes
    idx = 0
    for i in range(num_nodes):
        if i != node_id:
            other_params[idx] = get_model_params(nodes[i])
            idx += 1

    # Compute all differences at once
    diffs = get_diff_vectorized(curr_params, other_params)

    # Return mean difference
    return torch.mean(diffs)
