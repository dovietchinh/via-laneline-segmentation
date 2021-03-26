# via-laneline-segmentation

[English below]

repo này huấn luyện mạng phân đoạn vạch kẻ đường, được tích hợp trong dự án [via]()

## Công việc đã thực hiện

- [ ] triển khai mạng với các frame-work khác nhau

    - [x] Tensorflow - keras

    - [ ] Pytorch

    - [ ] MXnet

- [x] Cung cấp dữ liệu

- [x] Code Xử lý dữ liệu

- [x] Code augment dữ liệu

    -[x] RandAugment (chưa tối ưu)

- [x] Xây dựng mạng 

    - [x] U-net

    - [x] double U-net

- [x] Huấn luyện model 

    - [x] U-net

    - [ ] double U-net

- [x] Cung cấp pre-train model

- [x] Xây dựng metrics đánh giá

    - [x] DiceLoss

    - [x] Jaccard-index (hệ số iou)

- [ ] Demo kết quả 

    - [x] Demo kết quả trên ảnh

    - [ ] Demo kết quả trên video

- [ ] Tối ưu model và so sánh các kết quả

## Kết quả 

![demo1](images/demo1.png "demo")

## Cài đặt môi trường

- Cài đặt python >= 3.6

- Cài đặt thư viện :

    Các thư viện yêu cầu trong requirements.txt

    Các bạn mở terminal ,tạo môi trường mới, activate môi trường và cài các thư viện cần thiết. 

```

conda create -n lanlinesegment python==3.8

conda activate  lanelinesegment

pip install -r requirements.txt

```

## Cấu trúc thư mục
```

via-laneline-segmentation
├──data
|     ├── label_colors.txt
|     ├── train
│     |   ├── masks/*.png
│     |   ├── images/*.jpg
|     |   ├── new_masks/*.png
|     |   ├── new_images/*.jpg
|     |
|     ├── val
│     |   ├── masks/*.png
│     |   ├── images/*.jpg
|     |   ├── new_masks/*.png
|     |   ├── new_images/*.jpg
|     |
├── images                          # put images you want to test here
│     ├── demo_image_.jpg   
|    
├── src
|    ├── *.py
|
├── models                           # put pre-train models here
|    ├── *.h5

├── video                           # put videos your want to test here
|    ├── *.mp4
|
├── .gitignore
├── README.md
├── LICENSE
├── transform_data.py               # run this file first
├── demo_image.py
├── demo_video.py

```

## Các bước huấn luyện mạng

B1: Tải dữ liệu

B2: Xử lý dữ liệu

B3: Xây dựng mạng

B4: Viết Code augmenter

B5: Xây dựng DataSeuqence bao gồm augment và xử lý dữ liệu

B6: Viết metrics và hàm loss đánh giá

B7: Huấn luyện

B8: Chạy Demo

## Xử lý dữ liệu


dữ liệu gốc nhóm via cung cấp có độ phân giải 640x250, mask gồm 3 classes (Background,line,Road), để phù hợp với bài toán, tôi đọc ảnh và resize về  256x256 , ảnh mask chỉ giữ lại pixel line
tất cả được xử lý trong file tranform_data.py

các bạn có thể  tự lựa chọn resolution cho phù hợp, chỉnh sửa file configs/config,py

Sau khi chạy  

`python3 transform_data.py` 

trên terminal sẽ thu được new_masks và new_images trong folder data

Link dữ liệu gốc : [here]()
Link dữ liệu đã qua xử lys: [here]()


## Augment dữ liệu

Code augment dữ liệu trong file augmenter.py triển khai theo ý tưởng của bài báo RandAugment ở [8].

Để xem kết quả augments chạy terminal:
```
     cd **path/via-laneline-segmentation/

     python ./src/augmenter.py
     
```

Mỗi bức ảnh sẽ áp dụng theo chuỗi từ 1 -> N tranformation khác nhau với cường độ M.

Để sửa 2 tham số cho augmenter : chỉnh sửa trong file **config.py**

## Cấu trúc mạng

Tôi áp dụng mạng phổ biến nhất trong bài toán phân đoạn là mạng U-net và 1 phiên bản khác là double-Unet.

Tham khảo ở [1].

![demo4](images/demo4.png "demo")
![demo5](images/demo5.png "demo")

## Tham Khảo

[1] Double U-net: [DoubleU-Net: A Deep Convolutional Neural
Network for Medical Image Segmentation](https://arxiv.org/pdf/2006.04868.pdf)

[2] ASPP block :[DeepLab: Semantic Image Segmentation with
Deep Convolutional Nets, Atrous Convolution,
and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915v2.pdf)

[3] Squeeze-and-Excitation block: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

[4] [Repository 2020-CBMS-DoubleU-Net](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net)

[5] Data: [ISIC2018_task1 Lesion Boundary Segmentaion ](https://challenge2018.isic-archive.com/)

[6] data: [link here]()

[7] Pre-train model :[[link here]](https://drive.google.com/drive/folders/1cwNzf9OSG3PD_8MCeVobl04HystIbCSV?usp=sharing) 

[8] RandAugment paper : [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)


# Liên hệ

Thời gian chuẩn bị gấp rút nên có nhiều sai sót, mong nhận được ý kiến đóng góp từ các bạn.

 - **Email:** dovietchinh1998@gmail.com
 - **VNOpenAI team:** vnopenai@gmail.com
 - **facebook:**  https://www.facebook.com/profile.php?id=100005935236259

