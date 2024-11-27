import torch
import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# sam2 모델 체크포인트 경로
sam2_checkpoint = "C:/Users/SBA/finetuning/sam2/checkpoints/sam2.1_hiera_small.pt"

# model_config yaml파일
model_cfg = "sam2.1_hiera_s.yaml"

# build 패키지로 sam2 모델 가져오기
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)

# 원본 이미지 경로
og_data_dir = "D:/data_set/images"
# 마스크 이미지 경로
mask_data_dir = "D:/data_set/masks"

# for문으로 가져온 
image_names = [name for name in os.listdir(og_data_dir) if os.path.isfile(os.path.join(og_data_dir, name))]

data = [
    {
        'image': os.path.join(og_data_dir, image_name),
        'annotation': os.path.join(mask_data_dir, 'masks_' + image_name)
    }
    for image_name in image_names
]

def read_batch(data, batch_size=4):
    """Read a batch of images and their corresponding annotations."""
    batch_entries = np.random.choice(data, batch_size, replace=False)
    images, masks, points, labels = [], [], [], []
    
    for entry in batch_entries:
        image = cv2.imread(entry["image"])[..., ::-1]
        ann_map = cv2.imread(entry["annotation"])
        # print(image.shape,ann_map.shape)
        # Resize image and annotation
        r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
        image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

        # print(r)
        # print(image.shape,ann_map.shape)

        mat_map = ann_map[:, :, 0].astype(np.int64)
        ves_map = ann_map[:, :, 2].astype(np.int64)
        mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)

        inds = np.unique(mat_map)[1:]
        image_masks, image_points = [], []
        for ind in inds:
            mask = (mat_map == ind).astype(np.uint8)
            image_masks.append(mask)
            coords = np.argwhere(mask > 0)
            random_coord = np.array(coords[np.random.randint(len(coords))])
            image_points.append([[random_coord[1], random_coord[0]]])

        images.append(image)
        masks.append(np.array(image_masks))
        points.append(np.array(image_points))
        labels.append(np.ones([len(image_masks), 1]))

    return images, masks, points, labels


# Enable training for mask decoder and prompt encoder
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

# Set up optimizer and gradient scaler
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler()


# Training loop
best_iou = 0.0
batch_size = 128

for itr in range(50000):
    with torch.cuda.amp.autocast():
        images, masks, input_points, input_labels = read_batch(data, batch_size)
        
        batch_loss = 0
        batch_iou = 0
        
        for i in range(batch_size):
            if masks[i].shape[0] == 0:
                continue

            predictor.set_image(images[i])

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_points[i], input_labels[i], box=None, mask_logits=None, normalize_coords=True
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=True,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(masks[i].astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(dim=[1, 2])
            iou = inter / (gt_mask.sum(dim=[1, 2]) + (prd_mask > 0.5).sum(dim=[1, 2]) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

            loss = seg_loss + score_loss * 0.05
            batch_loss += loss
            batch_iou += np.mean(iou.cpu().detach().numpy())

        # Average loss and IoU over the batch
        batch_loss /= batch_size
        batch_iou /= batch_size

        # Backpropagation and optimization step
        predictor.model.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update and save model
        if itr == 0:
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * batch_iou

        if mean_iou > best_iou * 1.1:
            best_iou = mean_iou
            torch.save(predictor.model.state_dict(), f"model_batch.torch")
            print(f"Step {itr}, Accuracy (IoU) = {mean_iou:.4f}")


