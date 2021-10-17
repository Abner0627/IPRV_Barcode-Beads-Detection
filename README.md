# Barcode Beads Detection
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Barcode Beads Detection
│   main.py
│   func.py
│   config.py
│   requirements.txt  
│   README.md      
│       
└───img   
│   │     

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

3. 將欲處理的圖片放入`./img`中

4. 執行程式進行影像處理\
處理後的影像會生成至`./result`中，並以`result_原檔名.jpg`的形式命名。

## 方法說明
### 概述
```flow
st=>start: 輸入影像
ed=>end: 輸出影像
op1=>operation: Convert RGB to grayscale
op2=>operation: 2d convolution
op3=>operation: Adaptive thresholding
op4=>operation: Erosion
op5=>operation: Dilation
op6=>operation: Remove border

st->op1->op2->op3->op4->op5->op6->ed
```