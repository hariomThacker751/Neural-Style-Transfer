import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Avoid some windows OpenMP issues

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.choosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.choosen_features:
                features.append(x)
        return features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356

loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

def load_image(image_name):
   image = Image.open(image_name).convert('RGB')
   image = loader(image).unsqueeze(0)
   return image.to(device)

model = VGG().to(device).eval()

orignal_img = load_image('samples/content.png')
style_img = load_image('samples/style.png')

genrated = orignal_img.clone().requires_grad_(True)

total_steps = 200 # Reduced for quick evaluation
learning_rate = 0.05 # Increased learning rate a bit to see faster changes in 200 steps
alpha = 1
beta = 0.01

optimizer = optim.Adam([genrated], lr=learning_rate)

print(f"Starting Evaluation on {device} for {total_steps} steps...")
for step in range(total_steps + 1):
    generated_features = model(genrated)
    original_img_features = model(orignal_img)
    style_features = model(style_img)

    style_loss = 0
    original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features):

        batch_size, channel, height, width = gen_feature.shape

        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}/{total_steps}, Loss: {total_loss.item()}")
        save_image(genrated, "generated_eval.png")

print("Evaluation finished. Image saved as generated_eval.png.")
