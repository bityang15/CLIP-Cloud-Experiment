import importlib.metadata
import json
import platform
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torchvision.ops import nms


project_dir = Path('/root/autodl-tmp/clip-exp')
assert torch.cuda.is_available(), 'CUDA is unavailable in the experiment environment'
torch.manual_seed(1)

device = torch.device('cuda:0')
model = torch.nn.Conv2d(3, 8, kernel_size=3).to(device=device, dtype=torch.float16)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
images = torch.randn(4, 3, 32, 32, device=device, dtype=torch.float16)
loss = model(images).float().square().mean()
assert torch.isfinite(loss).item(), 'The FP16 forward pass produced a non-finite loss'
loss.backward()
assert all(
    parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
    for parameter in model.parameters()
), 'The FP16 backward pass produced missing or non-finite gradients'
optimizer.step()
assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
scores = torch.tensor([0.9, 0.8], device=device)
assert nms(boxes, scores, 0.5).tolist() == [0], 'The torchvision CUDA operator failed'
torch.cuda.synchronize()

package_names = [
    'torch', 'torchvision', 'numpy', 'scipy', 'scikit-learn', 'pandas',
    'pillow', 'matplotlib', 'ftfy', 'yacs', 'wilds', 'tensorboard', 'dassl',
]
report = {
    'checked_at': datetime.now().astimezone().isoformat(),
    'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in package_names},
    'pytorch_cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'fp16_forward_backward_and_sgd': 'passed',
    'torchvision_cuda_nms': 'passed',
    'synthetic_loss': loss.item(),
    'scope': 'Synthetic GPU operations and training-entry imports. Dataset training is a subsequent validation stage.',
}
report_path = project_dir / 'records' / 'environment_check.json'
report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
