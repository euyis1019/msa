import torch
import torch.nn.functional as F

def get_mma_loss(weight):
    '''
    MMA regularization in PyTorch
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss

def get_angular_loss_pytorch(weight):
    '''
    MMA regularization in PyTorch
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    theta = torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999))
    loss = -theta.mean()

    return loss

if __name__ == "__main__":
    # Test case 1: Dense layer weights
    weight1 = torch.randn(10, 5)
    mma_loss1 = get_mma_loss(weight1)
    angular_loss1 = get_angular_loss_pytorch(weight1)
    print(f"MMA Loss (Dense Layer): {mma_loss1.item()}")
    print(f"Angular Loss (Dense Layer): {angular_loss1.item()}")

    # Test case 2: Convolutional layer weights
    weight2 = torch.randn(10, 3, 3, 3)  # 10 filters, 3x3 kernel, 3 input channels
    mma_loss2 = get_mma_loss(weight2)
    angular_loss2 = get_angular_loss_pytorch(weight2)
    print(f"MMA Loss (Conv Layer): {mma_loss2.item()}")
    print(f"Angular Loss (Conv Layer): {angular_loss2.item()}")

    # Test case 3: Edge case with small weights
    weight3 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    mma_loss3 = get_mma_loss(weight3)
    angular_loss3 = get_angular_loss_pytorch(weight3)
    print(f"MMA Loss (Small Weights): {mma_loss3.item()}")
    print(f"Angular Loss (Small Weights): {angular_loss3.item()}")

    # Test case 4: Large weights
    weight4 = torch.randn(100, 50)
    mma_loss4 = get_mma_loss(weight4)
    angular_loss4 = get_angular_loss_pytorch(weight4)
    print(f"MMA Loss (Large Weights): {mma_loss4.item()}")
    print(f"Angular Loss (Large Weights): {angular_loss4.item()}")
