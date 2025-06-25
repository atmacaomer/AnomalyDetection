import kagglehub

# Download latest version
path = kagglehub.dataset_download("swaroopkml/cifar10-pngs-in-folders")

print("Path to dataset files:", path)