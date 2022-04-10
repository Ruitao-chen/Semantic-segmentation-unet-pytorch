"""
训练器模块
"""
import os
import unet
import torch
import dataset1
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 训练器
class Trainer:

    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.net = unet.UNet().to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.Adam(self.net.parameters())
        # 这里直接使用二分类交叉熵来训练，效果可能不那么好
        # 可以使用其他损失，比如DiceLoss、FocalLoss之类的
        self.loss_func = nn.BCELoss()
        # 设备好，batch_size和num_workers可以给大点
        self.loader = DataLoader(dataset1.Datasets(path), batch_size=4, shuffle=True, num_workers=0)

        # 判断是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No Param!")
        os.makedirs(img_save_path, exist_ok=True)


    def generate_matrix(self,gt_image, pre_image,num_class=8):
        gt_image=gt_image.cpu().detach().numpy()
        pre_image=pre_image.cpu().detach().numpy()
        mask = (gt_image >= 0) & (gt_image < num_class)#ground truth中所有正确(值在[0, classe_num])的像素label的mask
        
        label = num_class * gt_image[mask] + pre_image[mask] 
        label=label.astype(int)
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)#21 * 21(for pascal)
        return confusion_matrix

    def Mean_Intersection_over_Union(self,confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        MIoU = np.nanmean(MIoU) #跳过0值求mean,shape:[21]
        return MIoU
 
    # 训练
    def train(self, stop_value):
        epoch = 1
        while True:
            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                # 图片和分割标签
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 输出生成的图像
                out = self.net(inputs)
                loss = self.loss_func(out, labels)
                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                matrix =self.generate_matrix(y,x_)
                miou=self.Mean_Intersection_over_Union(matrix)

                img = torch.stack([x, x_, y], 0)
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))
                # print("image save successfully !")
            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}, mIOU: {miou}")
            torch.save(self.net.state_dict(), self.model)
            # print("model is saved !")

            # 备份
            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch >= stop_value:
                break
            epoch += 1


if __name__ == '__main__':
	# 路径改一下
    t = Trainer(r"E:/python/shendu/second/yuyifenge/msrc2_seg", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'./train_img')
    t.train(5)
