"""Local linear decoder Jacobians dx/dh estimated from samples."""

import torch


def local_decoder_jacobians(h, x, *, k=32, ridge=1e-3, chunk_size=256):
    """Estimate dx/dh at every sample by local ridge regression in h-space.

    For each anchor i, fit x_j - x_i ~= B_i (h_j - h_i) over the k nearest
    neighbors j of h_i (excluding i itself). Returns the stacked B_i with shape
    (num_anchors, x_dim, h_dim).
    """
    h = h.detach()
    x = x.detach()
    n, d_h = h.shape
    d_x = x.shape[-1]
    eye = torch.eye(d_h, device=h.device)
    jacobians = []

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        anchors_h = h[start:stop]
        anchors_x = x[start:stop]
        dist = torch.cdist(anchors_h, h)
        row = torch.arange(stop - start, device=h.device)
        dist[row, start + row] = float("inf")
        nn_idx = torch.topk(dist, k=min(k, n - 1), largest=False).indices

        dh = h[nn_idx] - anchors_h[:, None, :]
        dx = x[nn_idx] - anchors_x[:, None, :]
        gram = torch.einsum("bki,bkj->bij", dh, dh)
        cross = torch.einsum("bki,bkj->bij", dx, dh)
        coef_t = torch.linalg.solve(
            gram + ridge * eye.expand(stop - start, d_h, d_h),
            cross.transpose(-1, -2),
        )
        jacobians.append(coef_t.transpose(-1, -2).reshape(stop - start, d_x, d_h))

    return torch.cat(jacobians, dim=0)
