import torch
from torch import nn
import cv2
import numpy as np
import torch.fft
import math
import time
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from tqdm import trange

# ---------基本パラメータ設定---------
z = 400 #伝搬距離
batch_size = 2
valid_size = 100 # 検証データ枚数

pitch = 8.0e-3 # ピクセルピッチ
wavelength = 639e-6 # 光の波長
h = 544 # 画像の縦解像度
w = 960 # 画像の横解像度
slm_res = (h, w) # SLM解像度

# CPU/GPU自動切換え
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ---------伝搬フィルタ---------
# サンプルインデックスの作成(tesor型)
x = torch.linspace(-h//2, h//2-1, h, dtype=torch.float32)
y = torch.linspace(-w//2, w//2-1, w, dtype=torch.float32)
# tensor型に変換
h = np.array(h)
w = np.array(w)
h = torch.from_numpy(h)
w = torch.from_numpy(w)

# 周波数分解能
v = 1 / (h * pitch)
u = 1 / (w * pitch)
fx = x * v
fy = y * u

# 2D周波数平面
fX, fY = torch.meshgrid(fx, fy, indexing='ij')


# ---------角スペクトルの位相項---------
# 平方根中身
inside = 1.0 - (wavelength * fX)**2 - (wavelength * fY)**2
# 位相
H = (-1) * (2*np.pi/wavelength) * z * torch.sqrt(torch.clamp(inside, min=0.0))

# 伝搬可能領域のマスク(円)

prop_region = (inside >= 0.0).to(torch.float32)

Hreal = torch.cos(H) * prop_region # 実部
Himage = torch.sin(H) * prop_region # 虚部

# GPUに転送
H = torch.complex(Hreal, Himage).to(device)


# ---------U-Net---------
# Dounサンプリング
class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0)
        )
        
    def forward(self, x):
        out1=self.net1(x)
        out2=self.skip(x)
        out3=out1+out2
        out4=self.net2(out3)
        out5=out3+out4
        return out5
    
# Upサンプリング    
class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0)
        )
    
    def forward(self, x):
        out1=self.net1(x)
        out2=self.skip(x)
        out3=out1+out2
        out4=self.net2(out3)
        out5=out3+out4
        return out5

# HoloEncoder    
class holoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1=Down(1,16)
        self.netdown2=Down(16,32)
        self.netdown3=Down(32,64)
        self.netdown4=Down(64,96)
        self.netup0=Up(96,64)
        self.netup1=Up(64,32)
        self.netup2=Up(32,16)
        self.netup3=Up(16,1)
        self.tan=torch.nn.Hardtanh(-math.pi, math.pi) # 位相制限
        
    def forward(self, x):
        out1=self.netdown1(x)
        out2=self.netdown2(out1)
        out3=self.netdown3(out2)
        out4=self.netdown4(out3)
        
        # 各Upの入力にDownの出力を加算（スキップ接続）
        out5=self.netup0(out4)
        out6=self.netup1(out5+out3)
        out7=self.netup2(out6+out2)
        out8=self.netup3(out7+out1)
        out8=self.tan(out8)
        
        return out8

validpath=r"C:\Users\harap\Downloads\DIV2K_valid_HR"
model = holoencoder()
model.load_state_dict(torch.load('holoencoderstate.pth'))


if torch.cuda.is_available():
    model.cuda()


# ---------Transform---------
# 赤チャネルを取り出し，tensor変換transform⇒赤チャネルのみの（2160, 3840）テンソルにする
class ToTensorFromBGR(object):
    def __call__(self, img_bgr):
        r = img_bgr[..., 2].astype(np.float32) / 255.0 # 赤チャンネルを[0,1]へ
        tensor = torch.from_numpy(r).unsqueeze(0) # (1,1,H,W) バッチ次元、チャンネル数（赤）、高さ、幅
        return tensor

# Transforms定義
valid_transform = transforms.Compose([
    transforms.Lambda(lambda img: cv2.resize(img, (w, h))), # リサイズ
    ToTensorFromBGR(),
])

# Dataset定義
class DIV2KHoloDataset(torch.utils.data.Dataset):
    # インスタンス
    def __init__(self, root_dir, ids, transform=None):
        self.root_dir = root_dir
        self.ids = ids
        self.transform = transform
        
    # インデックス呼び出し処理
    def __getitem__(self, idx):
        c = self.ids[idx]
        imgpath = os.path.join(f"{self.root_dir}/{c:04d}.png")
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR) # OpenCVで読み込み
        
        if self.transform:
            img = self.transform(img)
            
        return img.to(device)
    
    # データセットサイズを返す処理
    def __len__(self):
        return len(self.ids)

valid_set = DIV2KHoloDataset(r"C:\Users\harap\Downloads\DIV2K_valid_HR",
                             ids=range(0,100),
                             transform=valid_transform)

valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


# ---------model load---------
model = holoencoder().to(device)
ckpt = "holoencoder.pth"
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state)
model.eval() # 検証

# ---------PSNR/SSIM---------
# PSNR定義    
# (B,H,W) or (H,W), 値域：[0,1]
def PSNR(p_img: torch.Tensor, t_img: torch.Tensor, data_range: float = 1.0, eps: float = 1e-10):
    # (B,H,W)の場合
    if p_img.dim() == 3:
        mse = torch.mean((p_img - t_img) ** 2, dim=(-2, -2)).clamp_min(eps) # (B,)ベクトル
        psnr = 20.0 * torch.log10(data_range / torch.sqrt(mse))
        return psnr.mean().item() # バッチごとに平均化・float型
    # (H,W)の場合    
    else:
        mse = torch.mean((p_img - t_img) ** 2).clamp_min(eps)
        return 20.0 * torch.log10(data_range / torch.sqrt(mse)).item()
    
# SSIM定義
def SSIM(p_img: torch.Tensor, t_img: torch.Tensor, data_range: float = 1.0):
    # skimageはnumpyなので都度CPUへ
    p = p_img.detach().cpu().numpy()
    t = t_img.detach().cpu().numpy()
    
    # (B,H,W)の場合
    if p.ndim == 3:
        s = 0.0
        for i in range(p.shape[0]):
            s += ssim(t[i], p[i], data_range=data_range)
        return s / p.shape[0] # バッチごとの平均を返す
    
    # (H,W)の場合    
    else:
        return ssim(t, p, data_range=data_range)
        
    
total_psnr = 0
total_ssim = 0
n_img = 0
save_dir = ".eval_vis"
os.makedirs(save_dir, exist_ok=True)


# ---------predict---------
# 予測フェーズ     
with torch.no_grad():
    for bidx, inputs_valid in tqdm(valid_loader):
        # GPUに転送
        inputs_valid = inputs_valid.to(device)
        
        # 位相マップを生成
        outputs_valid = model(inputs_valid)
        # (B, 1, H, W)⇒(B, H, W)
        outputs_valid = outputs_valid.squeeze(1)
        
        # 位相⇒複素波変換（後の計算のため）
        gray_real_valid = torch.cos(outputs_valid)
        gray_image_valid = torch.sin(outputs_valid)
        gray_valid = torch.complex(gray_real_valid, gray_image_valid)
       
        # 角スペクトル変換
        prop_valid = torch.fft.fftn(gray_valid, dim=(-2, -1)) # フーリエ変換
        prop_valid = H * prop_valid # 伝達関数
        prop_valid = torch.fft.ifftn(prop_valid, dim=(-2, -1)).abs() # 逆フーリエ変換→再構成像
        
        target = inputs_valid.squeeze(1)
        
        # 正規化
        maxv = torch.clamp(prop_valid.amax(dim=(-2,-1), keepdim=True), min=1e-8)
        prop_n = prop_valid / maxv
        
        # 指標
        total_psnr += PSNR(prop_n, target, data_range=1.0)
        total_ssim += SSIM(prop_n, target, data_range=1.0)
        n_img += 1
        
        # 可視化サンプル保存
        if bidx < 3:
            # 先頭１枚だけ保存
            r0 = (prop_n[0].detach().cpu().numpy() * 255.0).clip(0, 225).astype(np.unit8)
            t0 = (target[0].detach().cpu().numpy() * 255.0).clip(0, 225).astype(np.unit8)
            cv2.imwrite(os.path.join(save_dir, f"prop_b{bidx}.png"), r0)
            cv2.imwrite(os.path.join(save_dir, f"target_b{bidx}.png"), t0)
        
avg_psnr = total_psnr / n_img
avg_ssim = total_ssim / n_img
print(f"[VALID] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f} (N={n_img} batches)")