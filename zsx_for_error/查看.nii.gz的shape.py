from preprocess.common  import load_nii
img_example = r"H:\other_files\Heart_OpenDataset\OpenDataset\Training\Labeled\D3O9U9/D3O9U9_sa.nii.gz"
img, _, _ = load_nii(img_example)

print(img.shape)