import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
import numpy as np
from PIL import Image

class SimpleSAM:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_sam_model(model_path)
        self.model.to(self.device)
    
    def load_sam_model(self, model_path):
        # from segment_anything import sam_model_registry
        from ultralytics import SAM
        # Determine model type based on file name
        if 's' in model_path.lower():
            model_type = 'vit_s'
        elif 'b' in model_path.lower():
            model_type = 'vit_b'
        elif 'l' in model_path.lower():
            model_type = 'vit_l'
        else:
            model_type = 'vit_h'
        return SAM[model_type](checkpoint=model_path)

    class SAMDataset(Dataset):
        def __init__(self, img_dir, json_path):
            self.img_dir = img_dir
            
            # Load JSON annotations
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            
            # Validate that images exist
            self.valid_items = []
            for item in self.annotations:
                img_path = os.path.join(self.img_dir, item['image_filename'])
                if os.path.exists(img_path):
                    self.valid_items.append(item)
                else:
                    print(f"Warning: Image {item['image_filename']} not found")
            
        def __len__(self):
            return len(self.valid_items)
        
        def __getitem__(self, idx):
            item = self.valid_items[idx]
            
            # Load image
            img_path = os.path.join(self.img_dir, item['image_filename'])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image if needed (SAM typically expects 1024x1024)
            # image = cv2.resize(image, (1024, 1024))
            
            # Load mask from JSON annotations
            mask = self.create_mask_from_annotation(item['annotations'], image.shape[:2])
            
            # Convert to torch tensors
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).float()
            
            return {
                'image': image,
                'mask': mask,
                'image_path': img_path
            }
        
        def create_mask_from_annotation(self, annotations, shape):
            """Convert JSON annotations to binary mask"""
            mask = np.zeros(shape, dtype=np.float32)
            
            # This part depends on your JSON format
            # Example for polygon format:
            for ann in annotations:
                if 'segmentation' in ann:
                    # If polygons
                    if isinstance(ann['segmentation'], list):
                        poly = np.array(ann['segmentation']).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                    # If RLE
                    elif isinstance(ann['segmentation'], dict):
                        from pycocotools import mask as mask_utils
                        rle = ann['segmentation']
                        mask += mask_utils.decode(rle)
            
            return mask

    def train(self, data_dir, json_path, epochs=500, batch=16, name='sam_model'):
        """
        Train the SAM model
        Args:
            data_dir: Directory containing images
            json_path: Path to JSON file with annotations
            epochs: Number of training epochs
            batch: Batch size
            name: Name for saving the model
        """
        # Setup dataset and dataloader
        dataset = self.SAMDataset(data_dir, json_path)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Dataset size: {len(dataset)} images")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch_data in enumerate(dataloader):
                images = batch_data['image'].to(self.device)
                masks = batch_data['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, return_logits=True)
                loss = self._calculate_loss(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}")
            
            # Print epoch progress
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.save_model(f"{name}_epoch_{epoch+1}.pt")
    
    def _calculate_loss(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# Usage example:
if __name__ == "__main__":
    model = SimpleSAM("sam2_s.pt")
    model.train(
        data_dir="path/to/images",
        json_path="path/to/annotations.json",
        epochs=500,
        batch=8,
        name='test_train_1'
    )
