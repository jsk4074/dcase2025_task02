import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torcheval.metrics.functional import binary_auroc
from tqdm import tqdm
import numpy as np 

from data_prep.make_dataset import CustomDataset

torch.manual_seed(7777)
np.random.seed(7777)

def accuracy(predictions, labels):
    _, predicted_classes = torch.max(predictions, dim=1)  
    correct = (predicted_classes == labels).float()  
    acc = correct.sum() / len(correct) 
    return acc

def dataset_prep(
        path, 
        batch_size, 
        mode="test", 
        shuffle=False, 
        resize_shape=(128, 128),
        device = None,
    ):

    dataset = CustomDataset(
        dataset_root_path = path, 
        transform = None, 
        crop = False, 
        feature_extraction = False, 
        mode = mode,
        resize_shape = resize_shape, 
        device = device,
    )

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        # num_workers=1,  
        # persistent_workers=True,
    ) 

    return train_loader

def model_test( 
        batch_size, 
        dataset_path, 
        model_fe, 
        model_fcn, 
        device, 
        resize_shape = 128, 
        writer = None, 
        epoch = None, 
        train_embedding_feature = None, 
        train_embedding_label = None, 
    ): 
    print("============================ FCN (Discriminator) model TEST / Inference mode ============================") 

    # Load dataset
    train_loader = dataset_prep(
        dataset_path, 
        batch_size, 
        mode="test", 
        shuffle=False, 
        resize_shape=(resize_shape, resize_shape),
        device = device, 
    )

    # print("============================ Test / Validation mode ============================")
    total_correct = 0
    total_anomaly = 0
    total_samples = 0
    auc_score = 0

    model_prediction = []
    fc_prediction = []
    fc_confidence = []
    label_save = []
    save_loss = []
    embedding_feature_total = None

    with torch.no_grad():
        for img, label in tqdm(train_loader):
            # Make prediction for loss calculation
            img = img.to(device)
            
            # Fitting FCN 
            feature = model_fe(img)
            pred = model_fcn(feature)

            # Logging embeddings 
            if embedding_feature_total is None: 
                embedding_feature_total = feature
                if label.item() == 1: # Real anomaly label
                    embedding_label_total = torch.tensor([2])
                else: # Normal label
                    embedding_label_total = torch.tensor([0])
            else: 
                embedding_feature_total = torch.cat((embedding_feature_total, feature))
                if label.item() == 1: # Real anomaly label
                    embedding_label_total = torch.cat((embedding_label_total, torch.tensor([1])))
                else: # Normal label
                    embedding_label_total = torch.cat((embedding_label_total, torch.tensor([0])))

            confidence = torch.softmax(pred, dim=1)[:, 1].item()  # Extract probability of class 1
            predicted_class = pred.argmax().item()

            # Metric logging
            label_save.append(label.item())
            model_prediction.append(predicted_class)
            fc_confidence.append(confidence)  # Directly store confidence for class 1
            if predicted_class == label.item():
                total_correct += 1

            total_samples += 1

    embedding_feature_total = torch.cat((embedding_feature_total.detach().cpu(), train_embedding_feature.detach().cpu()), 0)
    embedding_label_total = torch.cat((embedding_label_total.detach().cpu(), train_embedding_label.detach().cpu()), 0)

    # Decoded embedding for EFA
    writer.add_embedding(
        embedding_feature_total,
        metadata = list(embedding_label_total),
        global_step = str(epoch),
    )

    # Accuracy calculation
    acc = total_correct / total_samples

    labels = torch.tensor(label_save, dtype=torch.int32)  
    scores = torch.tensor(fc_confidence, dtype=torch.float32)  

    auc_score = binary_auroc(scores, labels).item()
    abnormal_count = len([1 for i in model_prediction if i == 1])
    normal_count = len([0 for i in model_prediction if i == 0])

    writer.add_scalar("TEST auc", auc_score, epoch)
    writer.add_scalar("TEST acc", acc, epoch)
    print("test_roc_auc:", auc_score)
    print("test_acc:", acc * 100)
    print("Anomaly ratio:", abnormal_count)
    print("Normal ratio:", normal_count)
    print("="*60)

    return acc, auc_score, #visualization_dataframe



# print("=" * 25, "DEBUG", "=" * 25)