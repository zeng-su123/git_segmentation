import SimpleITK as sitk
from matplotlib import pyplot as plt


def showNii(img):
    for i in range(25):
        plt.imshow(img[i, 0, :, :], cmap='gray')
        plt.imsave("G:\image_save_cardiac2\second/"+str(i)+"_3.jpg",img[i, 1, :, :],cmap='gray') # 标签文件，分别是0，1，2，3
        # 所以图片是全黑的，因为像素值都很小
        # plt.show()


itk_img = sitk.ReadImage('H:\other_files\Heart_OpenDataset\OpenDataset\Training\Labeled\A0S9V9/A0S9V9_sa_gt.nii.gz')
img = sitk.GetArrayFromImage(itk_img)
print(type(img))
print(img.shape)  # (155, 240, 240) 表示各个维度的切片数量
showNii(img)
