import torch

def generate_narrowband_weights(M, N, phi, theta):
    theta = torch.deg2rad(theta)
    phi = torch.deg2rad(phi)

    m = torch.arange(M)
    n = torch.arange(N)

    # antenna arranged in grid, units of lambda
    x_pos, y_pos = torch.meshgrid(m, n, indexing='ij')
    x_pos = x_pos# - (M - 1) / 2
    y_pos = y_pos# - (N - 1) / 2

    spacing = 1 / 2 # array spacing assumed to be lambda/2

    x_delay = x_pos * spacing * torch.cos(phi) * torch.sin(theta) # delay along x
    y_delay = y_pos * spacing * torch.sin(phi) * torch.sin(theta) # delay along y

    delay_total = x_delay + y_delay # total delay
    delay_total = delay_total.flatten().unsqueeze(1)

    weights = torch.exp(2j * torch.pi * delay_total)

    return weights

def generate_narrowband_weights_azel(M, N, az, el):
    # convert azimuth and elevation to phi and theta
    # theta is elevation angle from z-axis, phi is rotation angle around z-axis
    # el is elevation above horizon, az is rotation around horizon
    az = torch.deg2rad(az)
    el = torch.deg2rad(el)
    # convert to point on sphere then convert back
    x = torch.cos(az) * torch.cos(el)
    y = torch.sin(az) * torch.cos(el)
    z = torch.sin(el)
    # rotate x y z by 90 degrees around y axis
    x, y, z = -z, -y, x
    phi = torch.atan2(y, x)
    theta = torch.atan2(z, torch.sqrt(x**2 + y**2))
    phi = -torch.rad2deg(phi)
    theta = torch.rad2deg(theta)
    theta = 90 - theta
    return generate_narrowband_weights(M, N, phi, theta)
