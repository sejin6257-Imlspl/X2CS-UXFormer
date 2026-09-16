import torch


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1,
                             horizontal_mean=None, vertical_mean=None, phi=None, theta=None):
    """
    구면 좌표계에서 카메라 위치를 샘플링 (gaussian 분포).
    phi/theta가 주어지면 그대로 사용(이 프로젝트는 항상 X-ray 실제 촬영 각도인 phi/theta를 명시적으로 전달).
    """
    import math
    horizontal_mean = math.pi * 0.5 if horizontal_mean is None else horizontal_mean
    vertical_mean = math.pi * 0.5 if vertical_mean is None else vertical_mean

    if phi is None or theta is None:
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r * torch.cos(phi)

    return output_points, phi, theta


def create_cam2world_matrix(forward_vector, origin, device=None):
    """카메라가 바라보는 방향과 원점으로부터 cam2world 4x4 변환 행렬 생성."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    return translation_matrix @ rotation_matrix