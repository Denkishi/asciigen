# ========================= IMPORT ========================= #
import sys, json
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QPixmap, QFont, QImage, QFontDatabase
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
            "invert": False,
            "dither": False,
            "edges": False,
            "mode": "levels",
            "text": "Powered by DENK",
            "levels": ["DDDENK", "FORZA SCIMMIE", "FULL FOCUS"],
            "level_modes": ["ascii", "ascii", "ascii"],
            "font_path": None,
            "char_set": "default",
            "edges_only": False,
            "edge_threshold": 50
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

def pixels_to_ascii(p, mode, text, levels, char_set="default", edges_only=False, edge_mask=None, level_modes=None):
    h,w = p.shape
    out=[]
    i=0
    chars = CHAR_SETS.get(char_set, CHAR_SETS["default"])
    if level_modes is None:
        level_modes = ["ascii", "ascii", "ascii"]
    for y in range(h):
        for x in range(w):
            px=int(p[y,x])
            if edges_only and edge_mask is not None:
                if edge_mask[y,x] <= 0:
                    out.append(" ")
                    continue
            if mode=="ascii":
                idx = px*(len(chars)-1)//255
                out.append(chars[idx])
            elif mode=="phrase":
                out.append(text[i%len(text)] if px<128 else " ")
                i+=1
            else:
                if px<85:
                    level_idx=0
                elif px<170:
                    level_idx=1
                else:
                    level_idx=2
                level_mode=level_modes[level_idx]
                if level_mode=="ascii":
                    idx = px*(len(chars)-1)//255
                    out.append(chars[idx])
                else:
                    s=levels[level_idx]
                    out.append(s[i%len(s)])
                    i+=1
        out.append("\n")
    return "".join(out)

# ========================= THREAD ========================= #
class RenderThread(QThread):
    finished = pyqtSignal(str, Image.Image)

    def __init__(self, img, params):
        super().__init__()
        self.img = img
        self.p = params

    def run(self):
        img = resize_image(self.img, self.p["width"]).convert("L")
        edge_mask = None
        if self.p["edges_only"]:
            p_orig = np.array(img)
            edge_mask = edge_detect(p_orig, self.p["edge_threshold"])
        img = ImageEnhance.Brightness(img).enhance(self.p["brightness"])
        img = ImageEnhance.Contrast(img).enhance(self.p["contrast"])
        p = np.array(img)
        if self.p["invert"]: p=255-p
        if self.p["edges"] and not self.p["edges_only"]: 
            p=edge_detect(p)
        if self.p["dither"]: p=dither(p)
        ascii_art = pixels_to_ascii(p,self.p["mode"],self.p["text"],self.p["levels"],self.p["char_set"],self.p["edges_only"],edge_mask,self.p.get("level_modes",["ascii","ascii","ascii"]))
        self.finished.emit(ascii_art,img)

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
        font_path=self.m.params["font_path"]
        font=None
        try:
            if font_path:
                import os
                if os.path.exists(font_path):
                    font=ImageFont.truetype(font_path,14)
                else:
                    db=QFontDatabase()
                    families=db.families()
                    if font_path in families:
                        styles=db.styles(font_path)
                        if styles:
                            try:
                                font_id=db.font(font_path,styles[0],12)
                                if hasattr(db,'fontFilePath'):
                                    path=db.fontFilePath(font_id)
                                    if path and os.path.exists(path):
                                        font=ImageFont.truetype(path,14)
                            except:
                                pass
                    if not font:
                        import platform
                        system=platform.system()
                        font_dirs=[]
                        if system=="Windows":
                            font_dirs=[os.path.join(os.environ.get("WINDIR","C:\\Windows"),"Fonts")]
                        elif system=="Darwin":
                            font_dirs=["/Library/Fonts","/System/Library/Fonts",os.path.expanduser("~/Library/Fonts")]
                        else:
                            font_dirs=["/usr/share/fonts","/usr/local/share/fonts",os.path.expanduser("~/.fonts")]
                        for font_dir in font_dirs:
                            if os.path.exists(font_dir):
                                for root,dirs,files in os.walk(font_dir):
                                    for file in files:
                                        if font_path.lower() in file.lower() and file.lower().endswith(('.ttf','.otf','.ttc')):
                                            try:
                                                font=ImageFont.truetype(os.path.join(root,file),14)
                                                break
                                            except:
                                                continue
                                    if font:
                                        break
                                if font:
                                    break
            if not font:
                font=ImageFont.load_default()
        except:
            font=ImageFont.load_default()
        lines=self.v.ascii.toPlainText().splitlines()
        w=max(len(l) for l in lines)*8; h=len(lines)*16
        img=Image.new("L",(w,h),255); d=ImageDraw.Draw(img)
        for i,l in enumerate(lines): d.text((0,i*16),l,font=font,fill=0)
        img.save(p)

    def export_svg(self):
        p,_=QFileDialog.getSaveFileName(self.v,"","ascii.svg","SVG (*.svg)")
        if not p: return
        with open(p,"w",encoding="utf8") as f:
            f.write(f"<svg xmlns='http://www.w3.org/2000/svg'><text font-family='monospace' font-size='12'>")
            y=14
            for l in self.v.ascii.toPlainText().splitlines():
                f.write(f"<tspan x='0' y='{y}'>{l}</tspan>")
                y+=14
            f.write("</text></svg>")

# ========================= VIEW ========================= #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ascii generator by DDDENK")
        self.model=AppModel()
        self.ctrl=Controller(self.model,self)
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
        v.addWidget(box)

        # TEXT
        box=Collapsible("Testo"); bl=QVBoxLayout(box.content)
        self.mode=QComboBox(); self.mode.addItems(["ascii","phrase","levels"])
        self.mode.setCurrentText("levels")
        self.mode.currentTextChanged.connect(lambda x:self.set("mode",x))
        self.text=QLineEdit(self.model.params["text"]); self.text.textChanged.connect(lambda x:self.set("text",x))
        bl.addWidget(QLabel("Livello 1"))
        level1_layout=QHBoxLayout()
        self.l1=QLineEdit(self.model.params["levels"][0])
        self.l1.textChanged.connect(lambda _:self.set("levels",[self.l1.text(),self.l2.text(),self.l3.text()]))
        self.mode_l1=QComboBox()
        self.mode_l1.addItems(["ascii","phrase"])
        self.mode_l1.setCurrentText(self.model.params.get("level_modes",["ascii","ascii","ascii"])[0])
        self.mode_l1.currentTextChanged.connect(lambda _:self.set("level_modes",[self.mode_l1.currentText(),self.mode_l2.currentText(),self.mode_l3.currentText()]))
        level1_layout.addWidget(self.l1)
        level1_layout.addWidget(self.mode_l1)
        bl.addLayout(level1_layout)
        bl.addWidget(QLabel("Livello 2"))
        level2_layout=QHBoxLayout()
        self.l2=QLineEdit(self.model.params["levels"][1])
        self.l2.textChanged.connect(lambda _:self.set("levels",[self.l1.text(),self.l2.text(),self.l3.text()]))
        self.mode_l2=QComboBox()
        self.mode_l2.addItems(["ascii","phrase"])
        self.mode_l2.setCurrentText(self.model.params.get("level_modes",["ascii","ascii","ascii"])[1])
        self.mode_l2.currentTextChanged.connect(lambda _:self.set("level_modes",[self.mode_l1.currentText(),self.mode_l2.currentText(),self.mode_l3.currentText()]))
        level2_layout.addWidget(self.l2)
        level2_layout.addWidget(self.mode_l2)
        bl.addLayout(level2_layout)
        bl.addWidget(QLabel("Livello 3"))
        level3_layout=QHBoxLayout()
        self.l3=QLineEdit(self.model.params["levels"][2])
        self.l3.textChanged.connect(lambda _:self.set("levels",[self.l1.text(),self.l2.text(),self.l3.text()]))
        self.mode_l3=QComboBox()
        self.mode_l3.addItems(["ascii","phrase"])
        self.mode_l3.setCurrentText(self.model.params.get("level_modes",["ascii","ascii","ascii"])[2])
        self.mode_l3.currentTextChanged.connect(lambda _:self.set("level_modes",[self.mode_l1.currentText(),self.mode_l2.currentText(),self.mode_l3.currentText()]))
        level3_layout.addWidget(self.l3)
        level3_layout.addWidget(self.mode_l3)
        bl.addLayout(level3_layout)
        bl.addWidget(self.mode)
        bl.addWidget(self.text)
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

    def open_image(self):
        p,_=QFileDialog.getOpenFileName(self,"","","Images (*.png *.jpg)")
        if p:
            self.model.image=Image.open(p)
            self.update_original_preview()
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

    def update_preview(self,ascii_art,img):
        self.ascii.setPlainText(ascii_art)
        try:
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
