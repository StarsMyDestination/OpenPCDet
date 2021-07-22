import torch
from pcdet.models.backbones_2d.map_to_rv.range_projection import SphereProjection


def test_sphere_projection():
    xyz = torch.rand(4, 3)
    print(f'x: \n {xyz}')
    rThetaPhi = SphereProjection.xyz_to_rThetaPhi(xyz)
    print(f'rThetaPhi: \n {rThetaPhi}')

    new_xyz = SphereProjection.rThetaPhi_to_xyz(rThetaPhi)
    print(f'new_xyz: \n {new_xyz}')

    assert torch.abs(xyz - new_xyz).sum() < 1e-5


if __name__ == '__main__':
    test_sphere_projection()
