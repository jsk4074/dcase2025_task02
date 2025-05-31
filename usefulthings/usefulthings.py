import torch
import matplotlib.pyplot as plt


def log_image(writer, tag, image_tensor, global_step, cmap=None):
    # image = image_tensor.detach().cpu()
    image = image_tensor.detach().cpu().mean(dim=0)
    
    height, width = image.shape[:2]
    dpi = 100  # adjust DPI as needed
    figsize = (width / dpi, height / dpi)

    # fig = plt.figure(figsize=figsize, dpi=dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # image = image.permute(1, 2, 0)

    ax.imshow(image, cmap=cmap)
    # ax.imshow(image)
    # ax.axis('off')  # Remove axes for a clean look.
    # Log the figure to TensorBoard.
    # writer.add_image(tag, image, global_step)
    writer.add_figure(tag, fig, global_step=global_step)
    
    # Close the figure to free up memory.
    plt.close(fig)

# Selects n random datapoints and their corresponding labels from a dataset
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def log_weight_histograms(writer, model, model_name, epoch):
    for name, param in model.named_parameters():

        if param.requires_grad:
            writer.add_histogram(f"{model_name}/{name}", param.data.cpu().numpy(), epoch)
