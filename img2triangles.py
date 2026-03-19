import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image
import os, sys

class TriangleRenderer(nn.Module):
    def __init__(self, num_triangles, img_size):
        super().__init__()
        self.num_triangles = num_triangles
        self.img_size = img_size # (h, w)
        
        # Initialize vertices randomly in [0, 1] range
        # Shape: (T, 3, 2) -> T triangles, 3 points each, (x, y) coordinates
        self.verts = nn.Parameter(torch.rand(num_triangles, 3, 2)*0.5 + 0.25)
        
        # Initialize RGB colors (No Alpha)
        self.colors = nn.Parameter(torch.rand(num_triangles, 3))
        
        # Z-order/Depth to determine which triangle is on top
        self.depth = nn.Parameter(torch.linspace(-1, 1, num_triangles))

    def forward(self, sharpness=100.0):
        h, w = self.img_size
        device = self.verts.device
        
        # Create coordinate grid
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        points = torch.stack([xx, yy], dim=-1).view(1, h, w, 2) # (1, H, W, 2)

        # Reshape vertices for broadcasting: (T, 1, 1, 3, 2)
        v = self.verts.view(self.num_triangles, 1, 1, 3, 2)
        
        # Differentiable Triangle Masking using cross-products
        # Edge vectors: (v1-v0), (v2-v1), (v0-v2)
        v0, v1, v2 = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        
        def edge_func(p, va, vb):
            return (p[..., 0] - va[..., 0]) * (vb[..., 1] - va[..., 1]) - \
                   (p[..., 1] - va[..., 1]) * (vb[..., 0] - va[..., 0])

        e0 = edge_func(points, v0, v1)
        e1 = edge_func(points, v1, v2)
        e2 = edge_func(points, v2, v0)

        # Soft-step function (sigmoid) creates the mask
        # Higher sharpness = crisper edges
        mask = torch.sigmoid(e0 * sharpness) * \
               torch.sigmoid(e1 * sharpness) * \
               torch.sigmoid(e2 * sharpness)
        mask = mask.unsqueeze(-1) # (T, H, W, 1)

        # Depth-based Compositing (Softmax-weighted average)
        # This allows gradients to flow through overlapping triangles
        #weights = F.softmax(self.depth.view(-1, 1, 1, 1) * 10.0 + mask.log().clamp(min=-10), dim=0)
        weights = F.softmax(self.depth.view(-1, 1, 1, 1) + torch.log(mask + 1e-6), dim=0)
        
        # Final image: Sum of (Triangle Color * Weight)
        rgb = (self.colors.view(self.num_triangles, 1, 1, 3) * weights).sum(dim=0)
        
        return rgb.permute(2, 0, 1) # (C, H, W)

def save_svg(verts, colors, img_size, filename):
    h, w = img_size
    with open(filename, 'w') as f:
        f.write(f'<svg width="{w}" height="{h}" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">\n')
        
        for i in range(len(verts)):
            v = verts[i]
            c = (colors[i] * 255).int().clamp(0, 255)
            color_str = f'rgb({c[0]},{c[1]},{c[2]})'
            points_str = " ".join([f"{p[0]},{p[1]}" for p in v])
            f.write(f'  <polygon points="{points_str}" fill="{color_str}" />\n')
        f.write('</svg>')

def train(target_path, out="output/", num_triangles=16, steps=1000, save_step=10, sharp_step=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prep image
    target_img = Image.open(target_path).convert("RGB")
    w, h = target_img.size
    target = transforms.ToTensor()(target_img).to(device)
    
    renderer = TriangleRenderer(num_triangles, (h, w)).to(device)
    optimizer = optim.Adam(renderer.parameters(), lr=0.01)

    output_dir = os.path.join(out, os.path.basename(target_path))
    os.makedirs(output_dir, exist_ok=True)

    for step in range(steps + 1):
        optimizer.zero_grad()
        
        # Higher sharpness over time for crisper triangles
        sharpness = min(100 + step * sharp_step, 1000)
        output = renderer(sharpness=sharpness)
        
        # 1. Reconstruction Loss (L1 is better for colors)
        loss_recon = F.l1_loss(output, target)

        # 2. Degeneracy Loss (Punish vertices getting too close)
        v = renderer.verts
        d01 = torch.norm(v[:, 0] - v[:, 1], dim=1)
        d12 = torch.norm(v[:, 1] - v[:, 2], dim=1)
        d20 = torch.norm(v[:, 2] - v[:, 0], dim=1)
        
        # Loss increases if distances are smaller than 0.05
        loss_size = torch.mean(1.0 / (d01 + 1e-4) + 1.0 / (d12 + 1e-4) + 1.0 / (d20 + 1e-4))
        
        # ???
        total_loss = loss_recon  + (loss_size * 0.001)
        
        total_loss.backward()
        optimizer.step()
        
        # Clamp vertices to [0, 1] range to keep them on canvas
        with torch.no_grad():
            renderer.verts.clamp_(0, 1)
            renderer.colors.clamp_(0, 1)

        if step % save_step == 0:
            print(f"Step {step} | Loss: {total_loss.item():.4f}")
            
            # Save SVG "Checkpoint"
            save_svg(
                renderer.verts.detach().cpu(),
                renderer.colors.detach().cpu(),
                (h, w),
                f"{output_dir}/step_{step:05d}.svg"
            )

            # Save Image
            utils.save_image(output, f"{output_dir}/step_{step:05d}.png")

if __name__ == "__main__":
    train(sys.argv[1])