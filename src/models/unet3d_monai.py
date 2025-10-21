from monai.networks.nets import UNet

def get_monai_unet(in_channels=1, out_channels=1, features=(16,32,64,128,256)):
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=features,
        strides=(2,2,2,2),
        num_res_units=2,
    )
    return model
