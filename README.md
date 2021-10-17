# Barcode Beads Detection
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Project
│   main.py
│   func.py
│   config.py
│   requirements.txt  
│   README.md      
│   ...    
└───img   
│   │   W_A1_0_3.jpg
│   │   ...
└───result   
│   │   result_W_A1_0_3.jpg
│   │   ...
└───ipynb 
```

## 前置工作
### 作業說明
* 目標\
透過影像處理的方式偵測圖中barcode beads的像素點。

### 環境
* python 3.8
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd [path/to/this/project]` 

2. 使用`pip install -r requirements.txt`安裝所需套件

3. 將欲處理的影像放入`./img`中

4. 執行程式進行影像處理\
`python main.py -I ./img/W_A1_0_3.jpg` 
其中`-I <image>`表示欲處理的影像路徑(相對路徑)與檔名。\
處理後的影像會生成至`./result`中，並以`result_原檔名.jpg`的形式命名；\
另可透過`-O <path>`改變輸出路徑(預設為`-O ./result`)。

## 方法說明
### 概述
![Imgur](https://i.imgur.com/OWtMeSx.png)

### 程式碼說明
#### Convert RGB to Grayscale
```py
# main.py
image_org = cv2.imread(args.image)
# 使用cv2讀取影像
image = func._rgb2gray(image_org)
# 使用_rgb2gray()轉換影像至灰階
```
```py
# func.py
def _rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return np.round(gray, 0)
# 將RGB影像以Gray = R*0.2989 + G*0.5870 + B*0.1140的方式進行轉換
```

#### 2d convolution
```py
# main.py
img_cv = func._conv2d(image, config.k_cv)
# 進行2d的convolution將影像平滑化
```
```py
# config.py
k_cv = np.ones((11, 11))
# convolution的kernel為11x11且數值全部為1之矩陣
```
```py
# func.py
def _conv2d(X, k):
    k_ = k / (k.shape[0] * k.shape[1])
    # 將kernel正規化
    X_pad = _pad(X, k_)
    # 對影像做padding，使convolution前後影像大小不變
    view_shape = tuple(np.subtract(X_pad.shape, k_.shape) + 1) + k_.shape
    # 計算convolution的視野大小(H', W', Hk, Wk) 
    # H' = H - (Hk - 1)，此處H'需等於H
    strides = X_pad.strides + X_pad.strides
    # 在W方向時，元素間隔皆為4 byte (X_pad[i, 0] to X_pad[i, 1])；
    # 在H方向時，元素間隔皆為4*W byte (X_pad[0, j] to X_pad[1, j])。
    # 由於前後兩個維度的計算方式一樣，故最終strides為(4W, 4, 4W, 4)    
    sub_matrices = as_strided(X_pad, view_shape, strides) 
    # 將X_pad依kernel大小分割並排成view_shape大小的矩陣
    cv = np.einsum('klij,ij->kl', sub_matrices, k_)
    # 矩陣內積 sub_matrices(S), k_(K), cv(C)
    # (C)_{kl} = (S)_{klij} · (K)_{ij}
    return cv  

def _pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    # 定義用padder的數值來做padding；預設為0
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def _pad(X, k, padder=0):
    XX_shape = tuple(np.subtract(X.shape, k.shape) + 1)
    # 計算convolution後的大小
    if XX_shape!=X.shape:
        P = np.subtract(X.shape, XX_shape) // 2
        # 計算需要padding多少
        if len(np.unique(P))!=1:
            print('kernel is not square matrix')
        else:
            X_ = np.pad(X, P[0], _pad_with, padder=padder)
    else:
        X_ = np.copy(X)
        # 當convolution後的大小與原本影像相同時，則僅作複製
    return X_    
```
假設原影像為3x3的矩陣，kernel為2x2，則sub_matrices示意圖如下(右)：
![Imgur](https://i.imgur.com/HB3MQKC.png)

#### Adaptive thresholding
```py
# main.py
img_adpt_thold = func._adpt_thold(img_cv, kernel_size=config.k_adpt_size, c=config.c)
# 進行adaptive thresholding分離前景與後景
```
```py
# config.py
k_adpt_size = 111
# 設定adaptive thresholding中的高斯kernel大小，此處為111
c = 3
# 參照cv2.adaptiveThreshold設定閥值之偏移量，此處為3
```
```py
# func.py
def _adpt_thold(image, kernel_size, c):
    k = gaussian_kernel(kernel_size)
    # 建立高斯kernel
    img_pad = _pad(image, k)
    view_shape = tuple(np.subtract(img_pad.shape, k.shape) + 1) + k.shape
    strides = img_pad.strides + img_pad.strides
    sub_matrices = as_strided(img_pad, view_shape, strides) 
    # 以上四步同_conv2d()
    thold = np.einsum('klij,ij->kl', sub_matrices, k) / np.sum(k) - c
    # 同樣進行內積計算閥值，並依偏移量調整
    img_adpt_thold = np.less(image, thold)
    # 設定小於閥值的像素為需要偵測的目標；即barcode beads處設定為1，其餘背景為0
    return img_adpt_thold

def gaussian_kernel(size):
    size = int(size) // 2
    # 以kernel大小的一半進行計算
    sigma = 0.3 * ((size*2 - 1) * 0.5 - 1) + 0.8
    # 計算高斯kernel的sigma值
    x, y = np.mgrid[-size:size+1, -size:size+1]
    # 預設xy方向的數值
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    # 計算二維高斯kernel
    return g  
```
二維高斯kernel計算方式：
![Imgur](https://i.imgur.com/FKAEBks.png)

#### Erosion
```py
# main.py
img_e = func._erosion(img_adpt_thold, config.k_e)
# 使用erosion去除邊緣
```
```py
# config.py
k_e = np.ones((5, 5))
# 使用數值全部為1的矩陣當作kernel，此處大小為5x5
```
```py
# func.py
def _erosion(X, k):
    X_pad = _pad(X, k)
    view_shape = tuple(np.subtract(X_pad.shape, k.shape) + 1) + k.shape
    strides = X_pad.strides + X_pad.strides
    sub_matrices = as_strided(X_pad, view_shape, strides) 
    cv = np.einsum('klij,ij->kl', sub_matrices, k)
    # 以上五步同_conv2d()
    return cv // (k.shape[0] * k.shape[1])
    # 當cv內元素除以kernel大小小於1時，表示該window內有包含背景，故定義為0；
    # 當cv內元素除以kernel大小大於1時，表示該window內只有前景，定義為1。
```

#### Dilation
```py
# main.py
img_d = func._dilation(img_e, config.k_d)
img_d = func._dilation(img_d, config.k_d)
img_d = func._dilation(img_d, config.k_d)
# 此處進行了三次dilation將影像邊緣擴增
```
```py
# config.py
k_d = np.ones((5, 5))
# 使用數值全部為1的矩陣當作kernel，此處大小為5x5
```
```py
# func.py
def _dilation(X, k):
    X = ~(X.astype(bool)) * 1
    # 將dilation理解成反過來對背景做erosion，
    # 故此處將前後景數值對換
    X_pad = _pad(X, k, padder=1)
    # 由於數值經過對換，故此處的padder為1
    view_shape = tuple(np.subtract(X_pad.shape, k.shape) + 1) + k.shape
    strides = X_pad.strides + X_pad.strides
    sub_matrices = as_strided(X_pad, view_shape, strides) 
    cv = np.einsum('klij,ij->kl', sub_matrices, k)
    cv_ = cv // (k.shape[0] * k.shape[1]) 
    # 以上五步同_conv2d()    
    return ~(cv_.astype(bool)) * 1
    # 最後將數值調換回來
```

#### Remove border
由於鏡頭邊框無法用上述方式去除，故此處僅能手動移除，如下圖示。
![Imgur](https://i.imgur.com/KJbAmmf.png)
```py
# main.py
final = func._rm(img_d)
# 移除邊框
```
```py
# func.py
def _rm(img):
    x = np.copy(img)
    x[3200:, :300] = 0
    # 定義H > 3200 & W < 300的範圍為鏡頭，將其定義為0
    return x
```

#### Image output
```py
result = (~final.astype(bool)) * 255
# 把bool形式的影像轉為0~255的陣列
fn = args.image.split('/')[-1]
# 擷取輸入影像的檔名
cv2.imwrite(os.path.join(args.output, 'result_' + fn), result)
# 使用cv2儲存影像，路徑由-O <path>決定，預測為./result
```

### 影像結果對比
W_A1_0_3
![Imgur](https://i.imgur.com/Apx0OKB.png)
W_A2_0_3
![Imgur](https://i.imgur.com/DvAsm7e.png)
W_A3_0_3
![Imgur](https://i.imgur.com/NwgiWzT.png)
由上圖可看出，越靠近中間，偵測的效果越好，其顆粒越大；\
然而越往外側顆粒點越小。\
同時為了偵測外側barcode beads，\
故此處erosion kernel較小，也造成影像中非顆粒點的地方也被偵測出來，\
因此之後實作或許可用CCL等方式篩選顆粒點大小，以過濾雜訊。