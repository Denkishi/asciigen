# ========================= IMPORT ========================= #
import sys, json
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QPixmap, QFont, QImage, QFontDatabase, QColor, QTextCharFormat, QFontMetrics
from PIL.ImageQt import ImageQt
import io
# ========================= MODEL ========================= #
class AppModel:
    def __init__(self):
        self.image = None
        self.params = {
            "width": 120,
            "brightness": 1.0,
            "contrast": 1.0,
            "invert": True,
            "dither": False,
            "edges": False,
            "mode": "levels",
            "text": "Powered by DENK",
            "levels": ["DDDENK", "FORZA SCIMMIE", "FULL FOCUS"],
            "level_modes": ["ascii", "ascii", "ascii"],
            "level_colors": [(0,0,0), (128,128,128), (255,255,255)],
            "font_path": None,
            "char_set": "default",
            "edges_only": False,
            "edge_threshold": 50,
            "color_mode": False,
            "color_limit": 16,
            "dominant_colors": [],
            "dominant_color_modes": [],
            "dominant_texts": [],
            "use_dominant_colors": False,
            "dominant_color_overrides": []
        }

# ========================= ASCII ENGINE ========================= #
ASCII_CHARS = "@%#*+=-:. "
CHAR_SETS = {
    "default": "@%#*+=-:. ",
    "numbers": "0123456789",
    "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "numbers_letters": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "symbols": "@%#*+=-:.!?/\\|<>[]{}()",
    "all": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@%#*+=-:.!?/\\|<>[]{}()"
}

def pixels_to_ascii(p, mode, text, levels, char_set="default", edges_only=False, edge_mask=None, level_modes=None, level_colors=None, color_mode=False, color_image=None, dominant_colors=None, dominant_color_modes=None, dominant_texts=None, use_dominant_colors=False, dominant_color_overrides=None):
    h,w = p.shape
    out=[]
    colors_out=[]
    i=0
    # Separate counters for each level to fix phrase mode
    num_levels = len(levels) if levels else 3
    level_counters = [0] * num_levels
    # Counters for dominant colors too
    dominant_counters = [0] * (len(dominant_colors) if dominant_colors else 0)
    chars = CHAR_SETS.get(char_set, CHAR_SETS["default"])
    if level_modes is None:
        level_modes = ["ascii", "ascii", "ascii"]
    if level_colors is None:
        level_colors = [(0,0,0), (128,128,128), (255,255,255)]
    
    def rgb_to_gray(r,g,b):
        return int(0.299*r + 0.587*g + 0.114*b)
    
    def find_closest_level(px_val, level_colors, is_grayscale):
        if is_grayscale:
            min_dist=float('inf')
            closest_idx=0
            for idx,color in enumerate(level_colors):
                gray_val=rgb_to_gray(color[0],color[1],color[2])
                dist=abs(px_val-gray_val)
                if dist<min_dist:
                    min_dist=dist
                    closest_idx=idx
            return closest_idx
        else:
            return 0
    
    is_grayscale=len(p.shape)==2
    num_levels=len(levels) if levels else len(level_colors)
    
    for y in range(h):
        for x in range(w):
            px=int(p[y,x])
            px_color=None
            if color_mode and color_image is not None:
                if len(color_image.shape)==3:
                    px_color=tuple(color_image[y,x])
                else:
                    px_color=(px,px,px)
            
            if edges_only and edge_mask is not None:
                if edge_mask[y,x] <= 0:
                    out.append(" ")
                    if color_mode:
                        colors_out.append(None)
                    continue
            
            if mode=="ascii":
                idx = px*(len(chars)-1)//255
                out.append(chars[idx])
                if color_mode:
                    colors_out.append(px_color)
            elif mode=="phrase":
                out.append(text[i%len(text)] if px<128 else " ")
                if color_mode:
                    colors_out.append(px_color if px<128 else None)
                i+=1
            elif mode == "colors":
                # Modalità "colors" - usa solo colori dominanti
                if use_dominant_colors and dominant_colors and color_mode and px_color:
                    # Combina colori base e colori dominanti
                    total_colors = level_colors + dominant_colors
                    total_num_levels = len(total_colors)
                    
                    # Trova il colore più vicino tra tutti i colori disponibili
                    min_dist=float('inf')
                    closest_idx=0
                    
                    for idx,color in enumerate(total_colors):
                        if idx>=total_num_levels:
                            break
                        dist_squared=(int(px_color[0])-int(color[0]))**2+(int(px_color[1])-int(color[1]))**2+(int(px_color[2])-int(color[2]))**2
                        if dist_squared<min_dist:
                            min_dist=dist_squared
                            closest_idx=idx
                    
                    level_idx=closest_idx
                    
                    # Determina la modalità corretta
                    if level_idx < len(level_colors):
                        # Livello base
                        if level_idx < len(level_modes):
                            level_mode = level_modes[level_idx]
                        else:
                            level_mode = "ascii"
                    else:
                        # Livello dominante
                        dominant_idx = level_idx - len(level_colors)
                        
                        # Default: usa logica normale dei livelli base
                        level_mode = "ascii"
                        if level_idx < len(level_modes):
                            level_mode = level_modes[level_idx]
                        
                        # Override: se abilitato e presente, usa la modalità del colore dominante
                        if (dominant_idx < len(dominant_color_overrides) and 
                            dominant_idx < len(dominant_color_modes) and
                            dominant_color_overrides[dominant_idx]):
                            level_mode = dominant_color_modes[dominant_idx]
                    
                    # Applica la modalità determinata
                    if level_mode=="ascii":
                        idx = px*(len(chars)-1)//255
                        out.append(chars[idx])
                        if color_mode:
                            colors_out.append(px_color)
                    else:  # phrase
                        if level_idx < len(levels):
                            s=levels[level_idx]
                            if not s or len(s)==0:
                                out.append(" ")
                            else:
                                # Use level-specific character index for proper phrase mode
                                level_counter = level_counters[level_idx] if level_idx < len(level_counters) else 0
                                out.append(s[level_counter % len(s)])
                                level_counters[level_idx] = level_counter + 1
                        else:
                            # Per colori dominanti, usa il testo specifico del colore dominante
                            if dominant_texts and dominant_idx < len(dominant_texts) and dominant_texts[dominant_idx]:
                                dominant_text = dominant_texts[dominant_idx]
                                dominant_counter = dominant_counters[dominant_idx] if dominant_idx < len(dominant_counters) else 0
                                out.append(dominant_text[dominant_counter % len(dominant_text)])
                                dominant_counters[dominant_idx] = dominant_counter + 1
                            else:
                                # Fallback al testo globale
                                out.append(text[i%len(text)])
                                i+=1
                        if color_mode:
                            colors_out.append(px_color)
                else:
                    # Fallback a ascii se non ci sono colori dominanti
                    idx = px*(len(chars)-1)//255
                    out.append(chars[idx])
                    if color_mode:
                        colors_out.append(px_color)
            else:
                # Modalità "levels" - comportamento normale
                if num_levels==0:
                    out.append(" ")
                    if color_mode:
                        colors_out.append(None)
                    continue
                
                if is_grayscale:
                    level_idx=find_closest_level(px,level_colors,True)
                else:
                    if color_mode and px_color:
                        min_dist=float('inf')
                        closest_idx=0
                        for idx,color in enumerate(level_colors):
                            if idx>=num_levels:
                                break
                            dist_squared=(int(px_color[0])-int(color[0]))**2+(int(px_color[1])-int(color[1]))**2+(int(px_color[2])-int(color[2]))**2
                        if dist_squared<min_dist:
                            min_dist=dist_squared
                            closest_idx=idx
                        level_idx=closest_idx
                    else:
                        if px<85:
                            level_idx=0
                        elif px<170:
                            level_idx=1
                        else:
                            level_idx=2
                        if level_idx>=num_levels:
                            level_idx=num_levels-1
                
                if level_idx>=len(level_modes):
                    level_mode="ascii"
                else:
                    level_mode=level_modes[level_idx]
                
                if level_mode=="ascii":
                    idx = px*(len(chars)-1)//255
                    out.append(chars[idx])
                    if color_mode:
                        colors_out.append(px_color)
                else:
                    if level_idx<len(levels):
                        s=levels[level_idx]
                        if not s or len(s)==0:
                            out.append(" ")
                        else:
                            out.append(s[i%len(s)])
                            i+=1
                    else:
                        out.append(" ")
                    if color_mode:
                        colors_out.append(px_color)
        out.append("\n")
        if color_mode:
            colors_out.append(None)
    
    result="".join(out)
    if color_mode:
        return result,colors_out
    return result

def resize_image(img, w):
    r = img.height / img.width
    return img.resize((w, int(w * r * 0.55)))

def edge_detect(p, threshold=50):
    gx = np.zeros_like(p)
    gy = np.zeros_like(p)
    gx[:,1:-1] = p[:,2:] - p[:,:-2]
    gy[1:-1,:] = p[2:,:] - p[:-2,:]
    edges = np.clip(np.sqrt(gx**2 + gy**2),0,255).astype(np.uint8)
    if threshold > 0:
        edges = np.where(edges > threshold, edges, 0)
    return edges

def dither(p):
    p = p.astype(np.float32)
    h,w = p.shape
    for y in range(h-1):
        for x in range(1,w-1):
            old = p[y,x]
            new = 255 if old>128 else 0
            err = old-new
            p[y,x]=new
            p[y,x+1]+=err*7/16
            p[y+1,x-1]+=err*3/16
            p[y+1,x]+=err*5/16
            p[y+1,x+1]+=err*1/16
    return np.clip(p,0,255).astype(np.uint8)

def extract_dominant_colors(image, n_colors):
    """Estrae i colori dominanti usando K-means clustering"""
    try:
        # Converti immagine in array numpy
        img_array = np.array(image)
        
        # Reshape per K-means (pixels x RGB)
        pixels = img_array.reshape(-1, 3)
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Ottieni i colori dei centroidi
        colors = kmeans.cluster_centers_.astype(int)
        
        # Ordina per frequenza (quanto appare ogni colore)
        labels = kmeans.labels_
        color_counts = []
        for i, color in enumerate(colors):
            count = np.sum(labels == i)
            color_counts.append((count, i))
        
        # Ordina per count decrescente
        color_counts.sort(reverse=True)
        
        # Ritorna colori ordinati per frequenza
        ordered_colors = [colors[idx] for count, idx in color_counts]
        return [tuple(color) for color in ordered_colors]
        
    except ImportError:
        # Fallback senza sklearn: usa PIL quantize
        try:
            # Quantizza l'immagine e ottieni la palette
            quantized = image.quantize(colors=n_colors)
            palette = quantized.getpalette()
            
            # Estrai i colori dalla palette
            colors = []
            for i in range(n_colors):
                idx = i * 3
                if idx + 2 < len(palette):
                    color = (palette[idx], palette[idx+1], palette[idx+2])
                    colors.append(color)
            
            return colors[:n_colors]
        except:
            # Fallback finale: colori di default
            default_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), 
                            (255,0,255), (0,255,255), (128,0,0), (0,128,0)]
            return default_colors[:n_colors]
    except Exception:
        # Fallback in caso di errore
        return [(128,128,128)] * n_colors

def pixels_to_ascii(p, mode, text, levels, char_set="default", edges_only=False, edge_mask=None, level_modes=None, level_colors=None, color_mode=False, color_image=None, dominant_colors=None, dominant_color_modes=None, dominant_texts=None, use_dominant_colors=False, dominant_color_overrides=None):
    h,w = p.shape
    out=[]
    colors_out=[]
    i=0
    # Separate counters for each level to fix phrase mode
    num_levels = len(levels) if levels else 3
    level_counters = [0] * num_levels
    # Counters for dominant colors too
    dominant_counters = [0] * (len(dominant_colors) if dominant_colors else 0)
    chars = CHAR_SETS.get(char_set, CHAR_SETS["default"])
    if level_modes is None:
        level_modes = ["ascii", "ascii", "ascii"]
    if level_colors is None:
        level_colors = [(0,0,0), (128,128,128), (255,255,255)]
    
    def rgb_to_gray(r,g,b):
        return int(0.299*r + 0.587*g + 0.114*b)
    
    def find_closest_level(px_val, level_colors, is_grayscale):
        if is_grayscale:
            min_dist=float('inf')
            closest_idx=0
            for idx,color in enumerate(level_colors):
                gray_val=rgb_to_gray(color[0],color[1],color[2])
                dist=abs(px_val-gray_val)
                if dist<min_dist:
                    min_dist=dist
                    closest_idx=idx
            return closest_idx
        else:
            return 0
    
    is_grayscale=len(p.shape)==2
    num_levels=len(levels) if levels else len(level_colors)
    
    for y in range(h):
        for x in range(w):
            px=int(p[y,x])
            px_color=None
            if color_mode and color_image is not None:
                if len(color_image.shape)==3:
                    px_color=tuple(color_image[y,x])
                else:
                    px_color=(px,px,px)
            
            if edges_only and edge_mask is not None:
                if edge_mask[y,x] <= 0:
                    out.append(" ")
                    if color_mode:
                        colors_out.append(None)
                    continue
            
            if mode=="ascii":
                idx = px*(len(chars)-1)//255
                out.append(chars[idx])
                if color_mode:
                    colors_out.append(px_color)
            elif mode=="phrase":
                out.append(text[i%len(text)] if px<128 else " ")
                if color_mode:
                    colors_out.append(px_color if px<128 else None)
                i+=1
            elif mode == "colors":
                # Modalità "colors" - usa solo colori dominanti
                if use_dominant_colors and dominant_colors and color_mode and px_color:
                    # Combina colori base e colori dominanti
                    total_colors = level_colors + dominant_colors
                    total_num_levels = len(total_colors)
                    
                    # Trova il colore più vicino tra tutti i colori disponibili
                    min_dist=float('inf')
                    closest_idx=0
                    
                    for idx,color in enumerate(total_colors):
                        if idx>=total_num_levels:
                            break
                        dist_squared=(int(px_color[0])-int(color[0]))**2+(int(px_color[1])-int(color[1]))**2+(int(px_color[2])-int(color[2]))**2
                        if dist_squared<min_dist:
                            min_dist=dist_squared
                            closest_idx=idx
                    
                    level_idx=closest_idx
                    
                    # Determina la modalità corretta
                    if level_idx < len(level_colors):
                        # Livello base
                        if level_idx < len(level_modes):
                            level_mode = level_modes[level_idx]
                        else:
                            level_mode = "ascii"
                    else:
                        # Livello dominante
                        dominant_idx = level_idx - len(level_colors)
                        
                        # Override: se abilitato e presente, usa la modalità del colore dominante
                        if (dominant_idx < len(dominant_color_overrides) and 
                            dominant_idx < len(dominant_color_modes) and
                            dominant_color_overrides[dominant_idx]):
                            level_mode = dominant_color_modes[dominant_idx]
                        else:
                            # Default: usa la modalità del livello base corrispondente
                            base_mode_idx = level_idx - len(level_colors)
                            if base_mode_idx < len(level_modes):
                                level_mode = level_modes[base_mode_idx]
                            else:
                                level_mode = "ascii"
                    
                    # Applica la modalità determinata
                    if level_mode=="ascii":
                        idx = px*(len(chars)-1)//255
                        out.append(chars[idx])
                        if color_mode:
                            colors_out.append(px_color)
                    else:  # phrase
                        # Usa i testi specifici per i colori dominanti
                        if level_idx < len(level_colors):
                            # Livello base - usa testo globale
                            out.append(text[i%len(text)])
                            i+=1
                        else:
                            # Colore dominante - usa testo specifico
                            dominant_idx = level_idx - len(level_colors)
                            if dominant_texts and dominant_idx < len(dominant_texts) and dominant_texts[dominant_idx]:
                                dominant_text = dominant_texts[dominant_idx]
                                dominant_counter = dominant_counters[dominant_idx] if dominant_idx < len(dominant_counters) else 0
                                out.append(dominant_text[dominant_counter % len(dominant_text)])
                                dominant_counters[dominant_idx] = dominant_counter + 1
                            else:
                                # Fallback al testo globale
                                out.append(text[i%len(text)])
                                i+=1
                        if color_mode:
                            colors_out.append(px_color)
                else:
                    # Fallback a ascii se non ci sono colori dominanti
                    idx = px*(len(chars)-1)//255
                    out.append(chars[idx])
                    if color_mode:
                        colors_out.append(px_color)
            else:
                # Modalità "levels" - comportamento normale
                if num_levels==0:
                    out.append(" ")
                    if color_mode:
                        colors_out.append(None)
                    continue
                
                if is_grayscale:
                    level_idx=find_closest_level(px,level_colors,True)
                else:
                    if color_mode and px_color:
                        min_dist=float('inf')
                        closest_idx=0
                        for idx,color in enumerate(level_colors):
                            if idx>=num_levels:
                                break
                            dist_squared=(int(px_color[0])-int(color[0]))**2+(int(px_color[1])-int(color[1]))**2+(int(px_color[2])-int(color[2]))**2
                            if dist_squared<min_dist:
                                min_dist=dist_squared
                                closest_idx=idx
                        level_idx=closest_idx
                    else:
                        if px<85:
                            level_idx=0
                        elif px<170:
                            level_idx=1
                        else:
                            level_idx=2
                        if level_idx>=num_levels:
                            level_idx=num_levels-1
                
                if level_idx>=len(level_modes):
                    level_mode="ascii"
                else:
                    level_mode=level_modes[level_idx]
                
                if level_mode=="ascii":
                    idx = px*(len(chars)-1)//255
                    out.append(chars[idx])
                    if color_mode:
                        colors_out.append(px_color)
                else:
                    if level_idx<len(levels):
                        s=levels[level_idx]
                        if not s or len(s)==0:
                            out.append(" ")
                        else:
                            out.append(s[i%len(s)])
                            i+=1
                    else:
                        out.append(" ")
                    if color_mode:
                        colors_out.append(px_color)
        out.append("\n")
        if color_mode:
            colors_out.append(None)
    
    result="".join(out)
    if color_mode:
        return result,colors_out
    return result

# ========================= THREAD ========================= #
class RenderThread(QThread):
    finished = pyqtSignal(str, Image.Image, object)

    def __init__(self, img, params):
        super().__init__()
        self.img = img
        self.p = params

    def run(self):
        color_mode = self.p.get("color_mode", False)
        if color_mode:
            img = resize_image(self.img, self.p["width"]).convert("RGB")
            color_limit = self.p.get("color_limit", 16)
            if color_limit < 256:
                img = img.quantize(colors=color_limit).convert("RGB")
            
            # Estrai colori dominanti se abilitato
            use_dominant_colors = self.p.get("use_dominant_colors", False)
            if use_dominant_colors:
                dominant_count = color_limit - 3  # base_levels = 3, senza limite massimo
                if dominant_count > 0:
                    try:
                        # Usa una versione ridotta dell'immagine per l'estrazione
                        img_small = resize_image(self.img, 100).convert("RGB")
                        dominant_colors = extract_dominant_colors(img_small, dominant_count)
                        self.p["dominant_colors"] = dominant_colors
                        
                        # Inizializza modalità e override per i nuovi colori
                        current_modes = self.p.get("dominant_color_modes", [])
                        while len(current_modes) < len(dominant_colors):
                            current_modes.append("ascii")
                        self.p["dominant_color_modes"] = current_modes[:len(dominant_colors)]
                        
                        current_overrides = self.p.get("dominant_color_overrides", [])
                        while len(current_overrides) < len(dominant_colors):
                            current_overrides.append(True)
                        self.p["dominant_color_overrides"] = current_overrides[:len(dominant_colors)]
                        
                    except Exception:
                        pass
        else:
            img = resize_image(self.img, self.p["width"]).convert("L")
        
        edge_mask = None
        if self.p["edges_only"]:
            if color_mode:
                p_orig = np.array(img.convert("L"))
            else:
                p_orig = np.array(img)
            edge_mask = edge_detect(p_orig, self.p["edge_threshold"])
        
        img = ImageEnhance.Brightness(img).enhance(self.p["brightness"])
        img = ImageEnhance.Contrast(img).enhance(self.p["contrast"])
        
        if color_mode:
            p = np.array(img.convert("L"))
            color_image = np.array(img)
        else:
            p = np.array(img)
            color_image = None
        
        if self.p["invert"]: p=255-p
        if self.p["edges"] and not self.p["edges_only"]: 
            p=edge_detect(p)
        if self.p["dither"]: p=dither(p)
        
        result = pixels_to_ascii(
            p, self.p["mode"], self.p["text"], self.p["levels"], 
            self.p["char_set"], self.p["edges_only"], edge_mask,
            self.p.get("level_modes",["ascii","ascii","ascii"]),
            self.p.get("level_colors",[(0,0,0),(128,128,128),(255,255,255)]),
            color_mode, color_image,
            self.p.get("dominant_colors", []),
            self.p.get("dominant_color_modes", []),
            self.p.get("dominant_texts", []),
            self.p.get("use_dominant_colors", False),
            self.p.get("dominant_color_overrides", [])
        )
        
        if color_mode:
            ascii_art, colors = result
        else:
            ascii_art = result
            colors = None
        
        self.finished.emit(ascii_art, img, colors)

# ========================= COLLAPSIBLE ========================= #
class Collapsible(QWidget):
    def __init__(self,title,tags=None):
        super().__init__()
        self.tags=tags or []
        self.btn=QToolButton(text=title,checkable=True,checked=True)
        self.btn.setArrowType(Qt.ArrowType.DownArrow)
        self.btn.clicked.connect(self.toggle)
        self.content=QWidget()
        self.anim=QPropertyAnimation(self.content,b"maximumHeight")
        self.anim.setDuration(200)
        self.tags_label=QLabel()
        self.tags_label.setStyleSheet("color: gray; font-size: 9px;")
        self.tags_label.setWordWrap(True)
        self.tags_label.hide()
        l=QVBoxLayout(self)
        l.addWidget(self.btn)
        l.addWidget(self.tags_label)
        l.addWidget(self.content)
    def toggle(self):
        c=self.btn.isChecked()
        self.btn.setArrowType(Qt.ArrowType.DownArrow if c else Qt.ArrowType.RightArrow)
        self.anim.setStartValue(self.content.maximumHeight())
        self.anim.setEndValue(self.content.sizeHint().height() if c else 0)
        self.anim.start()
        if self.tags:
            self.tags_label.setText(" • ".join(self.tags))
            self.tags_label.setVisible(not c)

# ========================= CONTROLLER ========================= #
class Controller:
    def __init__(self,model,view):
        self.m=model; self.v=view
        self.thread=None
        self.timer=QTimer(); self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.render)

    def schedule(self): self.timer.start(120)

    def render(self):
        if not self.m.image: return
        if self.thread and self.thread.isRunning():
            self.thread.quit(); self.thread.wait()
        img=self.m.image.copy()
        self.thread=RenderThread(img,self.m.params)
        self.thread.finished.connect(self.v.update_preview)
        self.thread.start()

    def export_txt(self):
        p,_=QFileDialog.getSaveFileName(self.v,"","ascii.txt","TXT (*.txt)")
        if p: open(p,"w",encoding="utf8").write(self.v.ascii.toPlainText())

    def export_png(self):
        p,_=QFileDialog.getSaveFileName(self.v,"","ascii.png","PNG (*.png)")
        if not p: return
        
        # Get the ASCII text from the preview
        ascii_text = self.v.ascii.toPlainText()
        if not ascii_text:
            return
        
        # Get font from the preview
        font = self.v.ascii.font()
        font_family = font.family()
        font_size = font.pointSize() if font.pointSize() > 0 else 12
        
        # Calculate dimensions based on actual preview
        lines = ascii_text.splitlines()
        if not lines:
            return
        
        # Use QFontMetrics to get accurate measurements like the preview
        fm = QFontMetrics(font)
        
        # Get precise character width for each character to avoid overlap
        def get_char_width(char_str):
            width = fm.horizontalAdvance(char_str)
            if width <= 0:
                width = fm.width(char_str)
            if width <= 0:
                width = 8
            return width
        
        # Get precise line height and spacing
        char_height = fm.height()
        if char_height <= 0:
            char_height = font_size + 4
        
        line_spacing = fm.lineSpacing()
        if line_spacing <= 0:
            line_spacing = char_height + 2
        
        # Calculate image dimensions with exact measurements per character
        max_line_width = 0
        for line in lines:
            line_width = 0
            for char in line:
                line_width += get_char_width(char)
            max_line_width = max(max_line_width, line_width)
        
        width = max_line_width
        height = len(lines) * line_spacing
        
        # Create image with transparent background
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        
        # Try to load the same font as preview with exact size
        pil_font = None
        try:
            pil_font = ImageFont.truetype(font_family, font_size)
        except:
            try:
                # Try common system fonts with exact size
                common_fonts = ["arial.ttf", "cour.ttf", "courier.ttf", "consola.ttf"]
                for font_name in common_fonts:
                    try:
                        pil_font = ImageFont.truetype(font_name, font_size)
                        break
                    except:
                        continue
            except:
                pass
        
        if pil_font is None:
            pil_font = ImageFont.load_default()
        
        # Get color mode and colors data
        color_mode = self.m.params.get("color_mode", False)
        colors_data = getattr(self.v, 'last_colors', None)
        
        # Draw text with exact spacing like preview
        char_idx = 0
        y_offset = 0  # Start from top, no padding
        
        for line_idx, line in enumerate(lines):
            x_offset = 0  # Start from left, no padding
            for char in line:
                # Check if we have color data for this character
                if (color_mode and colors_data and 
                    char_idx < len(colors_data) and 
                    colors_data[char_idx] is not None):
                    color = colors_data[char_idx]
                    fill_color = (color[0], color[1], color[2], 255)  # Add alpha
                else:
                    # Use the same text color as preview (black)
                    fill_color = (0, 0, 0, 255)  # Add alpha
                
                try:
                    d.text((x_offset, y_offset), char, font=pil_font, fill=fill_color)
                except Exception:
                    # Fallback if text rendering fails
                    d.text((x_offset, y_offset), char, font=ImageFont.load_default(), fill=fill_color)
                
                # Use individual character width to avoid overlap
                x_offset += get_char_width(char)
                char_idx += 1
            
            # Move to next line with exact spacing
            y_offset += line_spacing
            # Skip newline character in color data
            if (colors_data and char_idx < len(colors_data) and 
                colors_data[char_idx] is None):
                char_idx += 1
        
        # Save the image with high quality and transparency
        try:
            img.save(p, "PNG", compress_level=0)  # No compression for maximum quality
        except Exception:
            # Fallback without compression parameter
            img.save(p, "PNG")

    def export_svg(self):
        p,_=QFileDialog.getSaveFileName(self.v,"","ascii.svg","SVG (*.svg)")
        if not p: return
        lines=self.v.ascii.toPlainText().splitlines()
        if not lines:
            return
        
        color_mode=self.m.params.get("color_mode",False)
        colors_data=None
        if color_mode and hasattr(self.v,'last_colors'):
            colors_data=self.v.last_colors
        
        font_family="monospace"
        if self.m.params.get("font_path"):
            font_family=self.m.params["font_path"]
        
        max_line_len=max(len(l) for l in lines) if lines else 0
        char_width=8
        char_height=14
        w=max_line_len*char_width
        h=len(lines)*char_height
        
        with open(p,"w",encoding="utf8") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">')
            f.write(f'<text font-family="{font_family}" font-size="12">')
            char_idx=0
            for i,line in enumerate(lines):
                x=0
                for char in line:
                    if color_mode and colors_data and char_idx<len(colors_data) and colors_data[char_idx]:
                        color=colors_data[char_idx]
                        fill=f'fill="rgb({color[0]},{color[1]},{color[2]})"'
                    else:
                        fill='fill="black"'
                    f.write(f'<tspan x="{x}" y="{i*char_height+char_height}" {fill}>{char}</tspan>')
                    x+=char_width
                    char_idx+=1
                char_idx+=1
            f.write("</text></svg>")

# ========================= VIEW ========================= #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ascii generator by DDDENK")
        self.model=AppModel()
        self.ctrl=Controller(self.model,self)
        
        # Inizializza le variabili per i colori dominanti
        self.dominant_color_widgets = []
        
        self.build_ui()

    def build_ui(self):
        root=QWidget(); self.setCentralWidget(root)
        layout=QHBoxLayout(root)

        # PREVIEW
        self.original_image_lbl=QLabel()
        self.original_image_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_lbl.setMaximumSize(200,150)
        self.original_image_lbl.setMinimumSize(100,75)
        self.original_image_lbl.setScaledContents(True)
        self.original_image_lbl.setText("Originale")
        self.original_image_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        
        self.image_lbl=QLabel()
        self.image_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_lbl.setMaximumSize(300,300)
        self.image_lbl.setMinimumSize(150,150)
        self.image_lbl.setScaledContents(True)
        self.image_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        
        self.ascii=QTextEdit()
        if self.model.params["font_path"]:
            try:
                self.ascii.setFont(QFont(self.model.params["font_path"]))
            except:
                self.ascii.setFont(QFont("Courier"))
        else:
            self.ascii.setFont(QFont("Courier"))
        
        images_widget=QWidget()
        images_widget.setMinimumWidth(150)
        images_widget.setMaximumWidth(350)
        images_layout=QVBoxLayout(images_widget)
        images_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        images_layout.setSpacing(10)
        images_layout.setContentsMargins(5,5,5,5)
        images_layout.addWidget(self.original_image_lbl,0,Qt.AlignmentFlag.AlignCenter)
        images_layout.addWidget(self.image_lbl,0,Qt.AlignmentFlag.AlignCenter)
        images_layout.addStretch()
        
        preview_split=QSplitter(Qt.Orientation.Horizontal)
        preview_split.addWidget(images_widget)
        preview_split.addWidget(self.ascii)
        preview_split.setSizes([1,3])
        preview_split.setChildrenCollapsible(False)
        layout.addWidget(preview_split,3)

        # CONTROLS
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        side=QWidget(); v=QVBoxLayout(side)
        scroll.setWidget(side)
        layout.addWidget(scroll,1)

        def slider(lbl,min,max,val,cb,layout):
            s=QSlider(Qt.Orientation.Horizontal); s.setRange(min,max); s.setValue(val)
            s.valueChanged.connect(cb); layout.addWidget(QLabel(lbl)); layout.addWidget(s); return s

        # IMAGE
        box=Collapsible("Immagine")
        bl=QVBoxLayout(box.content)
        btn=QPushButton("Apri"); btn.clicked.connect(self.open_image)
        bl.addWidget(btn); v.addWidget(box)

        # BASE
        box=Collapsible("Base",["Larghezza","Luminosità","Contrasto","Inverti","Caratteri","Contorni"])
        bl=QVBoxLayout(box.content)
        self.w=slider("Larghezza",50,300,120,lambda x:self.set("width",x),bl)
        self.b=slider("Luminosità",50,200,100,lambda x:self.set("brightness",x/100),bl)
        self.c=slider("Contrasto",50,200,100,lambda x:self.set("contrast",x/100),bl)
        self.invert_cb=QCheckBox("Inverti colori")
        self.invert_cb.setChecked(self.model.params["invert"])
        self.invert_cb.toggled.connect(lambda x:self.set("invert",x))
        bl.addWidget(self.invert_cb)
        self.char_set_cb=QComboBox()
        self.char_set_cb.addItems(["default","numbers","letters","numbers_letters","symbols","all"])
        self.char_set_cb.setCurrentText(self.model.params["char_set"])
        self.char_set_cb.currentTextChanged.connect(lambda x:self.set("char_set",x))
        bl.addWidget(QLabel("Tipo caratteri"))
        bl.addWidget(self.char_set_cb)
        self.edges_only_cb=QCheckBox("Solo contorni")
        self.edges_only_cb.setChecked(self.model.params["edges_only"])
        self.edges_only_cb.toggled.connect(self.toggle_edges_only)
        bl.addWidget(self.edges_only_cb)
        self.edge_threshold_slider=slider("Threshold contorni",0,255,50,lambda x:self.set("edge_threshold",x),bl)
        self.edge_threshold_slider.setEnabled(self.model.params["edges_only"])
        self.color_mode_cb=QCheckBox("Modalità colori")
        self.color_mode_cb.setChecked(self.model.params.get("color_mode",False))
        self.color_mode_cb.toggled.connect(self.toggle_color_mode)
        bl.addWidget(self.color_mode_cb)
        # Container per slider e label
        color_limit_container=QWidget()
        color_limit_layout=QVBoxLayout(color_limit_container)
        color_limit_layout.setContentsMargins(0,0,0,0)
        
        self.color_limit_slider=slider("Limita colori",2,256,16,lambda x:self.update_color_limit(x),bl)
        self.color_limit_slider.setEnabled(self.model.params.get("color_mode",False))
        
        self.color_limit_label=QLabel(f"Colori limitati: {self.model.params.get('color_limit', 16)}")
        self.color_limit_label.setStyleSheet("color: gray; font-size: 11px; margin-left: 5px;")
        color_limit_layout.addWidget(self.color_limit_slider)
        color_limit_layout.addWidget(self.color_limit_label)
        bl.addWidget(color_limit_container)
        v.addWidget(box)

        # TEXT
        box=Collapsible("Testo"); bl=QVBoxLayout(box.content)
        self.mode=QComboBox(); self.mode.addItems(["ascii","phrase","levels","colors"])
        self.mode.setCurrentText("levels")
        self.mode.currentTextChanged.connect(lambda x:self.set("mode",x))
        self.text=QLineEdit(self.model.params["text"]); self.text.textChanged.connect(lambda x:self.set("text",x))
        bl.addWidget(self.mode)
        bl.addWidget(self.text)
        
        levels_widget=QWidget()
        self.levels_layout=QVBoxLayout(levels_widget)
        self.level_widgets=[]
        bl.addWidget(QLabel("Livelli:"))
        bl.addWidget(levels_widget)
        
        # I livelli principali rimangono solo 3
        for i in range(len(self.model.params["levels"])):
            self.add_level_widget(i)
        
        v.addWidget(box)

        # COLORI PRINCIPALI
        box=Collapsible("Colori Principali")
        bl=QVBoxLayout(box.content)
        
        self.dominant_colors_widget=QWidget()
        self.dominant_colors_layout=QVBoxLayout(self.dominant_colors_widget)
        self.dominant_color_widgets=[]
        bl.addWidget(self.dominant_colors_widget)
        
        # Inizializza i widget dei colori dominanti
        self.update_dominant_colors_ui()
        
        v.addWidget(box)

        # FONT
        box=Collapsible("Font",["Selettore font"])
        bl=QVBoxLayout(box.content)
        self.font_btn=QPushButton("Seleziona Font")
        self.font_btn.clicked.connect(self.select_font)
        bl.addWidget(self.font_btn)
        v.addWidget(box)

        # EXPORT
        box=Collapsible("Export"); bl=QVBoxLayout(box.content)
        for t,f in [("TXT",self.ctrl.export_txt),("PNG",self.ctrl.export_png),("SVG",self.ctrl.export_svg)]:
            b=QPushButton(t); b.clicked.connect(f); bl.addWidget(b)
        v.addWidget(box)

    def set(self,k,v):
        self.model.params[k]=v
        if k != "font_path":
            self.ctrl.schedule()
    
    def toggle_edges_only(self,checked):
        self.model.params["edges_only"]=checked
        self.edge_threshold_slider.setEnabled(checked)
        self.ctrl.schedule()
    
    def toggle_color_mode(self,checked):
        self.model.params["color_mode"]=checked
        self.color_limit_slider.setEnabled(checked)
        self.update_color_levels()
        self.ctrl.schedule()

    def update_color_limit(self, value):
        self.model.params["color_limit"] = value
        self.color_limit_label.setText(f"Colori limitati: {value}")
        if self.model.params.get("color_mode", False):
            self.update_dominant_colors()
        self.ctrl.schedule()
    
    def update_color_levels(self):
        """Aggiorna dinamicamente i livelli con colori dominanti"""
        if not self.model.params.get("color_mode", False):
            self.model.params["use_dominant_colors"] = False
            self.update_dominant_colors_ui()
            return
        
        color_limit = self.model.params.get("color_limit", 16)
        if color_limit <= 3:  # base_levels = 3
            self.model.params["use_dominant_colors"] = False
            self.update_dominant_colors_ui()
            return
        
        self.model.params["use_dominant_colors"] = True
        self.update_dominant_colors()

    def update_dominant_colors(self):
        """Estrae e aggiorna i colori dominanti"""
        color_limit = self.model.params.get("color_limit", 16)
        dominant_count = color_limit - 3  # base_levels = 3, senza limite massimo
        
        if self.model.image and dominant_count > 0:
            try:
                # Usa una versione ridotta dell'immagine per l'estrazione
                img_small = self.model.image.copy()
                img_small = img_small.resize((100, 100))
                if img_small.mode != "RGB":
                    img_small = img_small.convert("RGB")
                
                dominant_colors = extract_dominant_colors(img_small, dominant_count)
                self.model.params["dominant_colors"] = dominant_colors
                
                # Inizializza modalità e override per i nuovi colori
                current_modes = self.model.params.get("dominant_color_modes", [])
                while len(current_modes) < len(dominant_colors):
                    current_modes.append("ascii")
                self.model.params["dominant_color_modes"] = current_modes[:len(dominant_colors)]
                
                current_overrides = self.model.params.get("dominant_color_overrides", [])
                while len(current_overrides) < len(dominant_colors):
                    current_overrides.append(True)
                self.model.params["dominant_color_overrides"] = current_overrides[:len(dominant_colors)]
                
                self.update_dominant_colors_ui()
                
            except Exception:
                pass

    def update_dominant_colors_ui(self):
        """Aggiorna la UI dei colori dominanti"""
        # Rimuovi tutti i widget esistenti
        for widget_data in self.dominant_color_widgets:
            widget_data["widget"].setParent(None)
        self.dominant_color_widgets.clear()
        
        if not self.model.params.get("use_dominant_colors", False):
            return
        
        color_limit = self.model.params.get("color_limit", 16)
        dominant_count = color_limit - 3  # base_levels = 3, senza limite massimo
        dominant_colors = self.model.params.get("dominant_colors", [])
        dominant_modes = self.model.params.get("dominant_color_modes", [])
        dominant_overrides = self.model.params.get("dominant_color_overrides", [])
        
        for i in range(dominant_count):
            self.add_dominant_color_widget(i, dominant_colors, dominant_modes, dominant_overrides)

    def add_dominant_color_widget(self, idx, dominant_colors, dominant_modes, dominant_overrides):
        """Aggiunge un widget per un colore dominante"""
        color_widget=QWidget()
        color_layout=QHBoxLayout(color_widget)
        color_layout.setContentsMargins(0,0,0,0)
        
        # Indicatore colore
        if idx < len(dominant_colors):
            color = dominant_colors[idx]
            color_indicator=QLabel()
            color_indicator.setFixedSize(20, 20)
            color_indicator.setStyleSheet(f"""
                background-color: rgb({color[0]}, {color[1]}, {color[2]});
                border: 1px solid black;
                border-radius: 3px;
            """)
            color_indicator.setToolTip(f"Colore dominante: RGB({color[0]}, {color[1]}, {color[2]})")
            color_layout.addWidget(color_indicator)
        else:
            # Placeholder se non c'è colore
            placeholder=QLabel()
            placeholder.setFixedSize(20, 20)
            placeholder.setStyleSheet("background-color: gray; border: 1px solid black; border-radius: 3px;")
            color_layout.addWidget(placeholder)
        
        # Label del dominante
        label=QLabel(f"Dominante {idx+1}")
        label.setMinimumWidth(70)
        color_layout.addWidget(label)
        
        # Testo del dominante
        default_text = f"Colore {idx+1}"
        if idx < len(dominant_colors):
            level_text = self.model.params["levels"][3+idx] if 3+idx < len(self.model.params["levels"]) else default_text
        else:
            level_text = default_text
        
        text_edit=QLineEdit(level_text)
        text_edit.textChanged.connect(self.update_dominant_levels)
        color_layout.addWidget(text_edit)
        
        # Dropdown modalità
        mode_cb=QComboBox()
        mode_cb.addItems(["ascii","phrase"])
        
        if idx < len(dominant_modes):
            mode_cb.setCurrentText(dominant_modes[idx])
        else:
            mode_cb.setCurrentText("ascii")
        
        mode_cb.currentTextChanged.connect(self.update_dominant_levels)
        color_layout.addWidget(mode_cb)
        
        # Checkbox override
        override_cb=QCheckBox("Sovrascrivi")
        if idx < len(dominant_overrides):
            override_cb.setChecked(dominant_overrides[idx])
        else:
            override_cb.setChecked(True)
        
        override_cb.toggled.connect(self.update_dominant_levels)
        color_layout.addWidget(override_cb)
        
        # Salva i dati del widget
        widget_data={
            "widget":color_widget,
            "text":text_edit,
            "mode":mode_cb,
            "override":override_cb,
            "idx":idx
        }
        self.dominant_color_widgets.append(widget_data)
        self.dominant_colors_layout.addWidget(color_widget)

    def update_dominant_levels(self):
        """Aggiorna i dati dei colori dominanti"""
        texts=[]
        modes=[]
        overrides=[]
        
        for wd in self.dominant_color_widgets:
            texts.append(wd["text"].text())
            modes.append(wd["mode"].currentText())
            overrides.append(wd["override"].isChecked())
        
        self.model.params["dominant_color_modes"] = modes
        self.model.params["dominant_color_overrides"] = overrides
        
        
        
        # Aggiorna anche i livelli principali se necessario
        while len(self.model.params["levels"]) < len(texts) + 3:
            self.model.params["levels"].append("")
        for i, text in enumerate(texts):
            self.model.params["levels"][3+i] = text
        
        self.ctrl.schedule()
    
    def add_level_widget(self,idx):
        level_widget=QWidget()
        level_layout=QHBoxLayout(level_widget)
        level_layout.setContentsMargins(0,0,0,0)
        
        label=QLabel(f"Livello {idx+1}")
        label.setMinimumWidth(60)
        level_layout.addWidget(label)
        
        text_edit=QLineEdit(self.model.params["levels"][idx] if idx<len(self.model.params["levels"]) else "")
        text_edit.textChanged.connect(lambda: self.update_levels())
        level_layout.addWidget(text_edit)
        
        mode_cb=QComboBox()
        mode_cb.addItems(["ascii","phrase"])
        if idx<len(self.model.params["level_modes"]):
            mode_cb.setCurrentText(self.model.params["level_modes"][idx])
        mode_cb.currentTextChanged.connect(lambda: self.update_levels())
        level_layout.addWidget(mode_cb)
        
        
        
        
        
        widget_data={"widget":level_widget,"text":text_edit,"mode":mode_cb,"idx":idx}
        self.level_widgets.append(widget_data)
        self.levels_layout.addWidget(level_widget)
    
    
    
    
    
    def update_levels(self):
        levels=[]
        level_modes=[]
        for wd in self.level_widgets:
            levels.append(wd["text"].text())
            level_modes.append(wd["mode"].currentText())
        self.model.params["levels"]=levels
        self.model.params["level_modes"]=level_modes
        self.ctrl.schedule()

    def open_image(self):
        p,_=QFileDialog.getOpenFileName(self,"","","Images (*.png *.jpg)")
        if p:
            self.model.image=Image.open(p)
            self.update_original_preview()
            if self.model.params.get("color_mode", False):
                self.update_dominant_colors()
            self.ctrl.schedule()
    
    def update_original_preview(self):
        if self.model.image:
            try:
                img_copy=self.model.image.copy()
                if img_copy.mode == "L":
                    img_copy=img_copy.convert("RGB")
                elif img_copy.mode not in ("RGB","RGBA"):
                    img_copy=img_copy.convert("RGB")
                max_size=180
                w,h=img_copy.size
                if w>max_size or h>max_size:
                    ratio=min(max_size/w,max_size/h)
                    new_w=int(w*ratio)
                    new_h=int(h*ratio)
                    try:
                        img_copy=img_copy.resize((new_w,new_h),Image.Resampling.LANCZOS)
                    except AttributeError:
                        img_copy=img_copy.resize((new_w,new_h),Image.LANCZOS)
                buffer = io.BytesIO()
                img_copy.save(buffer, format='PNG')
                buffer.seek(0)
                pix = QPixmap()
                pix.loadFromData(buffer.getvalue())
                self.original_image_lbl.setPixmap(pix)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.original_image_lbl.setText("Errore caricamento")

    def select_font(self):
        current_font=QFont("Courier",10)
        if self.model.params["font_path"]:
            try:
                current_font=QFont(self.model.params["font_path"],10)
            except:
                pass
        font,ok=QFontDialog.getFont(current_font,self)
        if ok:
            family=font.family()
            self.model.params["font_path"]=family
            self.ascii.setFont(font)

    def update_preview(self,ascii_art,img,colors=None):
        self.last_colors=colors
        if colors:
            self.ascii.clear()
            cursor=self.ascii.textCursor()
            char_idx=0
            for char in ascii_art:
                if char=='\n':
                    cursor.insertText('\n')
                    while char_idx<len(colors) and colors[char_idx] is None:
                        char_idx+=1
                else:
                    if char_idx<len(colors) and colors[char_idx] is not None:
                        color=colors[char_idx]
                        fmt=QTextCharFormat()
                        fmt.setForeground(QColor(color[0],color[1],color[2]))
                        cursor.setCharFormat(fmt)
                        cursor.insertText(char)
                        fmt=QTextCharFormat()
                        cursor.setCharFormat(fmt)
                    else:
                        cursor.insertText(char)
                    char_idx+=1
        else:
            self.ascii.setPlainText(ascii_art)
        try:
            color_mode=self.model.params.get("color_mode",False)
            if color_mode:
                if img.mode == "L":
                    img_rgb = img.convert("RGB")
                elif img.mode not in ("RGB","RGBA"):
                    img_rgb = img.convert("RGB")
                else:
                    img_rgb = img
            else:
                if img.mode == "L":
                    img_rgb = img.convert("RGB")
                elif img.mode not in ("RGB","RGBA"):
                    img_rgb = img.convert("RGB")
                else:
                    img_rgb = img
            max_size=280
            w,h=img_rgb.size
            if w>max_size or h>max_size:
                ratio=min(max_size/w,max_size/h)
                new_w=int(w*ratio)
                new_h=int(h*ratio)
                try:
                    img_rgb=img_rgb.resize((new_w,new_h),Image.Resampling.LANCZOS)
                except AttributeError:
                    img_rgb=img_rgb.resize((new_w,new_h),Image.LANCZOS)
            buffer = io.BytesIO()
            img_rgb.save(buffer, format='PNG')
            buffer.seek(0)
            pix = QPixmap()
            pix.loadFromData(buffer.getvalue())
            self.image_lbl.setPixmap(pix)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.image_lbl.setText("Errore preview")


# ========================= MAIN ========================= #
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=MainWindow(); w.resize(1600,900); w.show()
    sys.exit(app.exec())
