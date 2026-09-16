import torchvision


def get_data_transform(in_channels, input_img_size):
    """X-ray/CT 이미지를 (resize + ImageNet 스타일 정규화)로 전처리."""
    data_aug = [torchvision.transforms.Resize(input_img_size)]
    if in_channels == 1:
        data_aug.append(torchvision.transforms.Normalize([0.485], [0.229]))
    elif in_channels == 3:
        data_aug.append(
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    return torchvision.transforms.Compose(data_aug)
