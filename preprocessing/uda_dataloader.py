import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def _load_txt_dataset(txt_file, image_size=224, root=""):
    """Load images and labels listed in a text file.

    Each line in the text file should contain an image path and the
    corresponding integer label separated by whitespace.
    Paths are considered relative to ``root`` if provided.
    Returns tensors ``data`` and ``targets`` suitable for FedD3 loaders.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    images = []
    labels = []
    with open(txt_file, "r") as f:
        for line in tqdm(f.readlines(), desc=f"loading {os.path.basename(txt_file)}"):
            path, label = line.strip().split()
            path = os.path.join(root, path)
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)
            labels.append(int(label))
    data = torch.stack(images)
    targets = torch.tensor(labels)
    return data, targets


def divide_domain_data(source_txts, target_txt, image_size=224, root=""):
    """Prepare federated splits for unsupervised domain adaptation.

    ``source_txts`` should be a list of domain text files used as sources.
    ``target_txt`` specifies the domain used for testing.
    Each source domain corresponds to a client containing all its samples.
    """
    trainset_config = {"users": [], "user_data": {}, "num_samples": []}

    for idx, txt in enumerate(source_txts):
        data, targets = _load_txt_dataset(txt, image_size=image_size, root=root)
        user = f"f_{idx:05d}"
        trainset_config["users"].append(user)
        trainset_config["user_data"][user] = {"x": data, "y": targets}
        trainset_config["num_samples"].append(len(targets))

    test_x, test_y = _load_txt_dataset(target_txt, image_size=image_size, root=root)
    test_data = {"x": test_x, "y": test_y}
    return trainset_config, test_data
