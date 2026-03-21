import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.utils.config import DOCS_DIR

def generate_architecture_diagram():
    os.makedirs(DOCS_DIR, exist_ok=True)
    out_path = os.path.join(DOCS_DIR, "system_architecture.png")
    
    components = [
        "Audio Input",
        "Audio Feature Extraction",
        "Deepfake Detection Model",
        "Speech-to-Text (Whisper)",
        "NLP Scam Detection",
        "Risk Scoring Engine",
        "Alert System",
        "Dashboard Interface"
    ]
    
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.axis('off')
    
    box_width = 0.6
    box_height = 0.08
    spacing = 0.12
    
    for i, label in enumerate(components):
        y_pos = 1.0 - (i * spacing) - box_height
        x_pos = 0.2
        
        # Draw box
        rect = patches.Rectangle((x_pos, y_pos), box_width, box_height, 
                                 linewidth=1.5, edgecolor='black', facecolor='lightblue', zorder=2)
        ax.add_patch(rect)
        
        # Add text
        plt.text(x_pos + box_width/2, y_pos + box_height/2, label,
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=12, weight='bold', zorder=3)
        
        # Draw arrow to the next box
        if i < len(components) - 1:
            arrow_y_start = y_pos
            arrow_y_end = y_pos - (spacing - box_height)
            ax.annotate('', xy=(0.5, arrow_y_end), xytext=(0.5, arrow_y_start),
                        arrowprops=dict(arrowstyle="->", lw=2), zorder=1)
            
    plt.title("System Architecture", fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Architecture diagram saved to {out_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
