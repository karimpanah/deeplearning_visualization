import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
import pandas as pd
import os
from matplotlib.animation import FuncAnimation, PillowWriter

class jd_tsne_cl:
    def __init__(self, model, device='cuda'):
        """
        jd_tsne_cl (Final Gold Edition):
        - Supports Auto-Extraction
        - Supports 2D, 3D, Heatmap, Misclassified, Interactive, GIF, 3-Views
        - Saves AND Shows plots inline
        - Includes Legends for 3-Views
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device).eval()
        
        self.features = []
        self.labels = []
        self.predictions = []
        self.confidences = []
        self.image_names = []
        self.tsne_2d = None
        self.tsne_3d = None
        
        # Custom and attractive color palette
        self.custom_colors = ['black', '#00008B', '#FF0000', '#006400', '#FFD700', 
                              '#FFA500', '#87CEEB', '#6B8E23', '#800080', '#8B4513', 
                              'gray', 'indigo', '#FF69B4', '#2E8B57', '#D2B48C']

    def _get_color_map(self, num_classes):
        if num_classes <= len(self.custom_colors): return self.custom_colors[:num_classes]
        else: return plt.cm.rainbow(np.linspace(0, 1, num_classes))

    def extract_and_compute(self, data_loader):
        self.features, self.labels, self.predictions, self.confidences, self.image_names = [], [], [], [], []
        print("⏳ [jd_tsne_cl] Extracting features..." )
        
        try:
            if hasattr(data_loader.dataset, 'samples'):
                self.image_names = [os.path.basename(s[0]) for s in data_loader.dataset.samples]
            else:
                self.image_names = [f"Img_{i}" for i in range(len(data_loader.dataset))]
        except: 
            self.image_names = [f"Img_{i}" for i in range(len(data_loader.dataset))]

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Extraction"):
                images = images.to(self.device)
                
                # A: Predict
                full_output = self.model(images)
                probs = F.softmax(full_output, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                self.confidences.append(max_probs.cpu()); self.predictions.append(preds.cpu())

                # B: Extract Logic (Smart Detection)
                if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'features'):
                    x = self.model.base_model.features(images)
                    x = F.adaptive_avg_pool2d(x, (1, 1)); x = torch.flatten(x, 1)
                elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'stage4'):
                     x = self.model.base_model.conv1(images); x = self.model.base_model.maxpool(x)
                     x = self.model.base_model.stage2(x); x = self.model.base_model.stage3(x)
                     x = self.model.base_model.stage4(x); x = self.model.base_model.conv5(x)
                     x = x.mean([2, 3])
                else:
                    if hasattr(self.model, 'features'): x = self.model.features(images); x = torch.mean(x, dim=[2, 3])
                    else: x = full_output

                self.features.append(x.cpu().reshape(x.size(0), -1))
                self.labels.append(labels.cpu())

        self.features = torch.cat(self.features).numpy()
        self.labels = torch.cat(self.labels).numpy()
        self.predictions = torch.cat(self.predictions).numpy()
        self.confidences = torch.cat(self.confidences).numpy()
        
        if len(self.image_names) != len(self.labels): 
            self.image_names = [f"Img_{i}" for i in range(len(self.labels))]

        print("⚙️ Computing t-SNE (2D & 3D)...")
        self.tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto').fit_transform(self.features)
        self.tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, init='pca', learning_rate='auto').fit_transform(self.features)

    # ==========================
    #  Main Plotting Method
    # ==========================
    def plot(self, class_names, modes=['all'], exclude=[], save_prefix='result'):
        if self.tsne_2d is None: raise ValueError("Run .extract_and_compute() first.")
        
        all_modes = ['2d', '3d', 'heatmap', 'mis', 'interactive', '3views', 'gif']
        final_modes = []
        if 'all' in modes: final_modes = all_modes.copy()
        else: final_modes = modes.copy()
            
        for item in exclude:
            if item in final_modes: final_modes.remove(item)
            
        print(f"🎨 Generating plots: {final_modes}")

        if '2d' in final_modes: self._plot_standard_2d(class_names, save_prefix)
        if '3d' in final_modes: self._plot_standard_3d(class_names, save_prefix)
        if 'heatmap' in final_modes: self._plot_density_heatmap(class_names, save_prefix)
        if 'mis' in final_modes: self._plot_misclassified(class_names, save_prefix)
        if 'interactive' in final_modes: self._plot_interactive(class_names, save_prefix)
        if '3views' in final_modes: self._plot_3views(class_names, save_prefix)
        if 'gif' in final_modes: self._generate_gif(class_names, save_prefix)

    # --- Internal Methods ---
    def _plot_interactive(self, class_names, save_prefix):
        df = pd.DataFrame({
            'x': self.tsne_2d[:, 0], 'y': self.tsne_2d[:, 1],
            'True Class': [class_names[i] for i in self.labels],
            'Predicted': [class_names[i] for i in self.predictions],
            'Confidence': [f"{c*100:.1f}%" for c in self.confidences],
            'Filename': self.image_names,
            'Status': ['Correct' if l==p else 'Misclassified' for l,p in zip(self.labels, self.predictions)]
        })
        color_map = {class_names[i]: c for i, c in enumerate(self._get_color_map(len(class_names)))}
        fig = px.scatter(df, x='x', y='y', color='True Class', symbol='Status',
                         hover_data=['Filename', 'Predicted', 'Confidence'], color_discrete_map=color_map,
                         title="Interactive t-SNE")
        fig.write_html(f"{save_prefix}_interactive.html")
        fig.show()

    def _plot_density_heatmap(self, class_names, save_prefix):
        df = pd.DataFrame({'x': self.tsne_2d[:,0], 'y': self.tsne_2d[:,1], 'Class': [class_names[i] for i in self.labels]})
        plt.figure(figsize=(10, 8))
        sns.kdeplot(data=df, x='x', y='y', hue='Class', fill=True, palette=self._get_color_map(len(class_names)), alpha=0.6)
        plt.title("Density Heatmap"); plt.tight_layout(); 
        plt.savefig(f"{save_prefix}_density.png"); plt.show()

    def _plot_misclassified(self, class_names, save_prefix):
        colors = self._get_color_map(len(class_names)); mis_mask = self.predictions != self.labels
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(self.tsne_2d[:, 0], self.tsne_2d[:, 1], c='lightgray', alpha=0.3)
        if np.sum(mis_mask) > 0:
            tsne_mis = self.tsne_2d[mis_mask]; labels_mis = self.labels[mis_mask]
            for i, cls_idx in enumerate(np.unique(self.labels)):
                sub_mask = labels_mis == cls_idx
                if np.sum(sub_mask) > 0:
                    c = colors[i] if isinstance(colors, list) else colors[i]
                    ax.scatter(tsne_mis[sub_mask, 0], tsne_mis[sub_mask, 1], c=c, marker='X', s=100, edgecolor='red', label=f"True: {class_names[cls_idx]}")
            ax.legend(); ax.set_title(f"Misclassified ({np.sum(mis_mask)})")
        else: ax.set_title("No Errors!")
        plt.tight_layout(); plt.savefig(f"{save_prefix}_mis.png"); plt.show()

    def _plot_standard_2d(self, class_names, save_prefix):
        unique_cls = np.unique(self.labels); colors = self._get_color_map(len(class_names))
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls_idx in enumerate(unique_cls):
            mask = self.labels == cls_idx; c = colors[i] if isinstance(colors, list) else colors[i]
            label_txt = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            ax.scatter(self.tsne_2d[mask, 0], self.tsne_2d[mask, 1], c=c, label=label_txt, alpha=0.7, s=40)
        ax.legend(); ax.set_title("2D t-SNE"); plt.tight_layout(); 
        plt.savefig(f"{save_prefix}_2d.png"); plt.show()

    def _plot_standard_3d(self, class_names, save_prefix):
        unique_cls = np.unique(self.labels); colors = self._get_color_map(len(class_names))
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
        for i, cls_idx in enumerate(unique_cls):
            mask = self.labels == cls_idx; c = colors[i] if isinstance(colors, list) else colors[i]
            label_txt = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            ax.scatter(self.tsne_3d[mask, 0], self.tsne_3d[mask, 1], self.tsne_3d[mask, 2], c=c, label=label_txt, s=40)
        ax.legend(); ax.set_title("3D t-SNE"); plt.tight_layout(); 
        plt.savefig(f"{save_prefix}_3d.png"); plt.show()

    def _plot_3views(self, class_names, save_prefix):
        # Modified version with Legend
        unique_cls = np.unique(self.labels); colors = self._get_color_map(len(class_names))
        fig = plt.figure(figsize=(18, 6)); views = [("Top", 90, -90), ("Front", 0, 0), ("Side", 0, -90)]
        for k, (t, el, az) in enumerate(views):
            ax = fig.add_subplot(1, 3, k+1, projection='3d')
            for i, cls_idx in enumerate(unique_cls):
                mask = self.labels == cls_idx; c = colors[i] if isinstance(colors, list) else colors[i]
                label_txt = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
                ax.scatter(self.tsne_3d[mask, 0], self.tsne_3d[mask, 1], self.tsne_3d[mask, 2], c=c, s=20, alpha=0.6, label=label_txt)
            ax.view_init(elev=el, azim=az); ax.set_title(t)
            ax.legend(loc='best', fontsize='small') # Added Legend
        plt.tight_layout(); plt.savefig(f"{save_prefix}_3views.png"); plt.show()

    def _generate_gif(self, class_names, save_prefix):
        unique_cls = np.unique(self.labels); colors = self._get_color_map(len(class_names))
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
        scatters = []
        for i, cls_idx in enumerate(unique_cls):
            mask = self.labels == cls_idx; c = colors[i] if isinstance(colors, list) else colors[i]
            sc = ax.scatter(self.tsne_3d[mask, 0], self.tsne_3d[mask, 1], self.tsne_3d[mask, 2], c=c, s=40)
            scatters.append(sc)
        def update(frame): ax.view_init(elev=20, azim=frame); return scatters
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=50)
        ani.save(f"{save_prefix}_rotation.gif", writer=PillowWriter(fps=20)); plt.close()
        print(f"✅ GIF Saved: {save_prefix}_rotation.gif")
