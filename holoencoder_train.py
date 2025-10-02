import torch
from torch import nn, optim
import cv2
import numpy as np
import torch.fft
import math
from scipy import io
from tqdm import tqdm
from tqdm import trange # プログレスバー付きループ
# import time
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

# ---------基本パラメータ設定---------
z = 400 #伝搬距離
num_epoch = 30 # 学習エポック数
train_size = 700 # 学習データ枚数
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
    
    
# ---------Training Parameters---------
lr = 0.001 # 学習率
batch_size = 2 # バッチサイズ
model = holoencoder() # モデル設定
criterion = nn.MSELoss() # 損失関数設定
# optimizier = torch.optim.Adam(model.parameters(), lr=lr)

# GPUに回す
if torch.cuda.is_available():
    model.to(device)
    
H=H.to(device)
optvers=[{'params': model.parameters()}]
optimizier = torch.optim.Adam(optvers, lr=lr)
train_path=r"C:\Users\harap\Downloads\DIV2K_train_HR"
valid_path=r"C:\Users\harap\Downloads\DIV2K_valid_HR"
train_loss=[] # 学習損失の推移
valid_loss=[] # 検証損失推移


# ---------Transform---------
# 赤チャネルを取り出し，tensor変換transform⇒赤チャネルのみの（2160, 3840）テンソルにする
class ToTensorFromBGR(object):
    def __call__(self, img_bgr):
        r = img_bgr[..., 2].astype(np.float32) / 255.0 # 赤チャンネルを[0,1]へ
        tensor = torch.from_numpy(r).unsqueeze(0) # (1,1,H,W) バッチ次元、チャンネル数（赤）、高さ、幅
        return tensor

# Transforms定義
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: cv2.resize(img, (960, 544))), # リサイズ
    ToTensorFromBGR(),
])

valid_transform = transforms.Compose([
    transforms.Lambda(lambda img: cv2.resize(img, (960, 544))), # リサイズ
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

train_set = DIV2KHoloDataset(r"C:\Users\harap\Downloads\DIV2K_train_HR",
                            ids=range(100,800),
                            transform=train_transform)

valid_set = DIV2KHoloDataset(r"C:\Users\harap\Downloads\DIV2K_valid_HR",
                             ids=range(0,100),
                             transform=valid_transform)

target_amp = train_set[0].to(device)
print(target_amp.shape, target_amp.min().item(), target_amp.max().item())
# → torch.Size([1,2160,3840]) 0.0 1.0

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

print(len(train_loader))

# for images, labels in train_loader:
#     break

# print(images.shape)
# print(labels.shape)
# 乱数固定
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# 評価結果記録用
history = np.zeros((0,5))


# ---------Training Loop---------
# 繰り返しメインループ
for epoch in range(num_epoch):
    # 1エポックあたりの累積損失（平均化前）
    train_current_loss, valid_current_loss = 0, 0
    # 1エポックあたりのデータ累積件数
    n_train, n_valid = 0, 0
    
    # 訓練フェーズ
    for inputs in tqdm(train_loader):
        # GPUに転送
        inputs = inputs.to(device)
        
        # 位相マップを生成
        outputs = model(inputs)
        # (B, 1, H, W)⇒(B, H, W) 2つ目のちゃんねる消去
        outputs = outputs.squeeze(1)
        
        # 位相⇒複素波変換（後の計算のため）
        gray_real = torch.cos(outputs)
        gray_image = torch.sin(outputs)
        gray = torch.complex(gray_real, gray_image)
       
        # 角スペクトル変換
        prop = torch.fft.fftn(gray, dim=(-2, -1)) # フーリエ変換
        prop = prop * H # 伝達関数
        prop = torch.fft.ifftn(prop, dim=(-2, -1)).abs() # 逆フーリエ変換→再構成像
        
        # inputs = inputs.double()
        target = inputs.squeeze(1)
        # 損失計算
        loss = criterion(prop, target)
        # 勾配の初期化
        optimizier.zero_grad()
        # 勾配計算
        loss.backward()
        # パラメータ更新
        optimizier.step()
        # 損失合計
        train_current_loss += loss.item()
   
# 予測フェーズ     
with torch.no_grad():
    for inputs_valid in tqdm(valid_loader):
        # GPUに転送
        inputs_valid = inputs_valid.to(device)
        
        # 位相マップを生成
        outputs_valid = model(inputs_valid)
        # (1, 1, H, W)⇒(H, W)
        outputs_valid = outputs_valid.squeeze(1)
        
        # 位相⇒複素波変換（後の計算のため）
        gray_real_valid = torch.cos(outputs_valid)
        gray_image_valid = torch.sin(outputs_valid)
        gray_valid = torch.complex(gray_real_valid, gray_image_valid)
       
        # 角スペクトル変換
        prop_valid = torch.fft.fftn(gray_valid, dim=(-2, -1)) # フーリエ変換
        prop_valid = H * prop_valid # 伝達関数
        prop_valid = torch.fft.ifftn(prop_valid, dim=(-2, -1)).abs() # 逆フーリエ変換→再構成像
        
        # inputs = inputs.double()
        target_valid = inputs_valid.squeeze(1)
        # 損失計算
        loss_valid = criterion(prop_valid, target_valid)
        # 損失合計
        valid_current_loss += loss_valid.item()

    # 学習後の保存
    ave_train_loss = train_current_loss / n_train
    ave_valid_loss = valid_current_loss / n_valid
    # 表示
    print(f'Epoch [{epoch+1}/{epoch}], loss: {ave_train_loss:.5f} val_loss: {ave_valid_loss:.5f}')
    # 記録
    item = np.array([epoch+1, ave_train_loss, ave_valid_loss]) 
    history = np.vstack((history, item))   
    torch.save(model.state_dict(), 'holoencoderstate.pth')