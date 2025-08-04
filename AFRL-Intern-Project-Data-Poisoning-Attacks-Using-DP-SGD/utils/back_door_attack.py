
def apply_backdoor_attack(dataset, trigger_label=0, poison_fraction=0.01, patch_size=3):
    num_poison = int(len(dataset) * poison_fraction)
    indices = torch.randperm(len(dataset))[:num_poison]

    for idx in indices:
        image, _ = dataset[idx]
        image[0, 0:patch_size, 0:patch_size] = 1.0  # white square patch (top-left)
        dataset.data[idx] = (image * 255).byte()
        dataset.targets[idx] = trigger_label  # all patched images labeled as `trigger_label`

    return dataset
    