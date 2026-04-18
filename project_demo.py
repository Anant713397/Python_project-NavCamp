import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageFont
import numpy as np
from scipy import ndimage
import heapq
import math

# ─────────────────────────────────────────────
#  ONLY ONE IMAGE NEEDED — the red lines map
#  User sees this map with green path on top
# ─────────────────────────────────────────────
MAP_PATH = r"C:\Users\LOQ\OneDrive\Pictures\Camera Roll\Campus map-new.jpeg"

# ─────────────────────────────────────────────
#  BUILDINGS — give me coords from Paint
#  and I'll update these
# ─────────────────────────────────────────────
BUILDINGS = {
    "Hubble":                               (140, 673),
    "Academic Block (Block 11)":            (860, 581),
    "Management Block":                     (461, 107),
    "Basketball Court":                     (155, 173),
    "Kandoli Mess":                         (132, 330),
    "Library":                              (240, 261),
    "IT Tower":                             (344, 350),
    "School of Design":                     (378, 351),
    "Block - 6":                            (289, 352),
    "Block - 7":                            (428, 414),
    "Mac (Upper)":                          (88,  447),
    "Mac (Lower)":                          (69,  473),
    "Day Care Centre":                      (532, 455),
    "Laboratory Block":                     (805, 434),
    "Block - 1":                            (152, 511),
    "Block - 2":                            (235, 518),
    "Block - 3":                            (334, 513),
    "Block - 4":                            (247, 561),
    "Block - 5":                            (366, 603),
    "School of Computer Science (Block 9)": (642, 517),
    "Tulip":                                (880, 581),
    "Block - 10":                           (718, 669),
    "Food Court":                           (343, 666),
    "Gate (Entry)":                         (363,  63),
}

# ─────────────────────────────────────────────
#  BUILD WALKABLE MASK
# ─────────────────────────────────────────────
def build_walkable(img):
    data = np.array(img, dtype=np.float32)
    r,g,b = data[:,:,0],data[:,:,1],data[:,:,2]
    is_red  = (r>130)&(g<80)&(b<80)&(r>g*1.8)&(r>b*1.8)
    is_blue = (b>130)&(r<150)&(g<180)&(b>r+30)

    # Connect each blue dot to nearest red pixel
    red_ys,red_xs = np.where(is_red)
    labeled,n = ndimage.label(is_blue)
    blues = []
    for i in range(1,n+1):
        ys,xs = np.where(labeled==i)
        if 8<=len(xs)<=300:
            blues.append((int(xs.mean()),int(ys.mean())))

    conn = Image.fromarray(np.zeros((img.height,img.width),dtype=np.uint8))
    cd   = ImageDraw.Draw(conn)
    for bx,by in blues:
        dists = (red_ys-by)**2+(red_xs-bx)**2
        idx   = np.argmin(dists)
        cd.line([(bx,by),(int(red_xs[idx]),int(red_ys[idx]))],fill=255,width=4)

    # Bridge small gaps between disconnected components
    combined_temp = is_red|np.array(conn)>0|is_blue
    wt = ndimage.binary_dilation(combined_temp,iterations=4)
    labeled2,n2 = ndimage.label(wt)
    sizes = {i:np.sum(labeled2==i) for i in range(1,n2+1)}
    comps = [k for k,v in sorted(sizes.items(),key=lambda x:-x[1])[:8] if v>100]
    for i in range(len(comps)):
        for j in range(i+1,len(comps)):
            c1,c2 = comps[i],comps[j]
            c1_ys,c1_xs = np.where(labeled2==c1)
            c2_ys,c2_xs = np.where(labeled2==c2)
            min_d=float('inf'); best=None
            for k in range(0,len(c1_xs),15):
                dists=(c2_ys-c1_ys[k])**2+(c2_xs-c1_xs[k])**2
                idx=np.argmin(dists)
                d=math.sqrt(dists[idx])
                if d<min_d: min_d=d; best=(int(c1_xs[k]),int(c1_ys[k]),int(c2_xs[idx]),int(c2_ys[idx]))
            if min_d<50:
                cd.line([(best[0],best[1]),(best[2],best[3])],fill=255,width=6)

    connectors = np.array(conn)>0
    walkable   = ndimage.binary_dilation(is_red|connectors|is_blue,iterations=4)
    return walkable

# ─────────────────────────────────────────────
#  SNAP
# ─────────────────────────────────────────────
def snap(walkable,x,y):
    x,y = int(x),int(y)
    h,w = walkable.shape
    if 0<=y<h and 0<=x<w and walkable[y,x]: return x,y
    ys,xs = np.where(walkable)
    idx = np.argmin((ys-y)**2+(xs-x)**2)
    return int(xs[idx]),int(ys[idx])

# ─────────────────────────────────────────────
#  A*
# ─────────────────────────────────────────────
def astar(walkable,start,goal):
    h,w   = walkable.shape
    sx,sy = snap(walkable,start[0],start[1])
    gx,gy = snap(walkable,goal[0],goal[1])
    heap=[(0.0,(sy,sx))]; came_from={}; g_score={(sy,sx):0.0}
    DIRS=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    COSTS=[1,1,1,1,1.414,1.414,1.414,1.414]
    goal_n=(gy,gx)
    while heap:
        _,cur=heapq.heappop(heap)
        if cur==goal_n:
            path=[]
            while cur in came_from:
                path.append((cur[1],cur[0])); cur=came_from[cur]
            path.append((sx,sy)); path.reverse(); return path
        cy,cx=cur
        for (dy,dx),cost in zip(DIRS,COSTS):
            ny,nx=cy+dy,cx+dx
            if 0<=ny<h and 0<=nx<w and walkable[ny,nx]:
                ng=g_score[cur]+cost; nb=(ny,nx)
                if ng<g_score.get(nb,float('inf')):
                    g_score[nb]=ng
                    heapq.heappush(heap,(ng+math.hypot(ny-gy,nx-gx),nb))
                    came_from[nb]=cur
    return None

# ─────────────────────────────────────────────
#  RENDER — green path on red lines map
# ─────────────────────────────────────────────
def render_map(base_img,path=None,src_name=None,dst_name=None):
    img  = base_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font_bold = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf",14)
    except:
        font_bold = ImageFont.load_default()

    if path and len(path)>1:
        pts = path[::3]+[path[-1]]
        for i in range(len(pts)-1):
            draw.line([pts[i],pts[i+1]],fill=(0,0,0,80),width=9)
        for i in range(len(pts)-1):
            draw.line([pts[i],pts[i+1]],fill=(0,220,0,255),width=5)

        sx,sy=path[0]
        draw.ellipse([sx-10,sy-10,sx+10,sy+10],fill=(0,255,0,255),outline=(255,255,255,255),width=2)
        if src_name:
            bbox=draw.textbbox((sx+13,sy-8),src_name,font=font_bold)
            draw.rectangle([bbox[0]-3,bbox[1]-3,bbox[2]+3,bbox[3]+3],fill=(255,255,255,220))
            draw.text((sx+13,sy-8),src_name,fill=(0,150,0),font=font_bold)

        ex,ey=path[-1]
        draw.ellipse([ex-10,ey-10,ex+10,ey+10],fill=(255,50,50,255),outline=(255,255,255,255),width=2)
        if dst_name:
            bbox=draw.textbbox((ex+13,ey-8),dst_name,font=font_bold)
            draw.rectangle([bbox[0]-3,bbox[1]-3,bbox[2]+3,bbox[3]+3],fill=(255,255,255,220))
            draw.text((ex+13,ey-8),dst_name,fill=(200,0,0),font=font_bold)

    return img.convert("RGB")

# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
class CampusMapApp:
    def __init__(self,root):
        self.root=root
        self.root.title("UPES Campus Navigator")
        self.root.configure(bg="#1e1e2e")

        print("Loading map...")
        self.base_img = Image.open(MAP_PATH).convert("RGB")
        self.walkable = build_walkable(self.base_img)
        print(f"Ready! Walkable px: {self.walkable.sum()}")

        self._build_ui()
        self._refresh_map()

    def _build_ui(self):
        style=ttk.Style(); style.theme_use("clam")
        style.configure("TLabel",background="#1e1e2e",foreground="#cdd6f4",font=("Segoe UI",11))
        style.configure("TButton",background="#89b4fa",foreground="#1e1e2e",font=("Segoe UI",11,"bold"),padding=6)
        style.configure("TCombobox",font=("Segoe UI",11))

        top=tk.Frame(self.root,bg="#1e1e2e",pady=10)
        top.pack(fill="x",padx=20)

        tk.Label(top,text="🏫 UPES Campus Navigator",bg="#1e1e2e",fg="#89b4fa",
                 font=("Segoe UI",16,"bold")).pack(pady=(0,10))

        row=tk.Frame(top,bg="#1e1e2e"); row.pack()
        names=sorted(BUILDINGS.keys())

        ttk.Label(row,text="📍 From:").grid(row=0,column=0,padx=8,sticky="e")
        self.src_var=tk.StringVar()
        ttk.Combobox(row,textvariable=self.src_var,values=names,width=30,state="readonly").grid(row=0,column=1,padx=8)

        ttk.Label(row,text="🎯 To:").grid(row=0,column=2,padx=8,sticky="e")
        self.dst_var=tk.StringVar()
        ttk.Combobox(row,textvariable=self.dst_var,values=names,width=30,state="readonly").grid(row=0,column=3,padx=8)

        btns=tk.Frame(top,bg="#1e1e2e"); btns.pack(pady=8)
        ttk.Button(btns,text="🔍 Find Path",command=self.find_path).pack(side="left",padx=8)
        ttk.Button(btns,text="🗑 Clear",command=self.clear_path).pack(side="left",padx=8)

        self.status=tk.Label(top,text="Select source and destination.",
                             bg="#1e1e2e",fg="#a6e3a1",font=("Segoe UI",10,"italic"))
        self.status.pack()

        frame=tk.Frame(self.root,bg="#1e1e2e")
        frame.pack(fill="both",expand=True,padx=10,pady=(0,10))
        self.canvas=tk.Canvas(frame,bg="#1e1e2e",bd=0,highlightthickness=0)
        hbar=tk.Scrollbar(frame,orient="horizontal",command=self.canvas.xview)
        vbar=tk.Scrollbar(frame,orient="vertical",command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set,yscrollcommand=vbar.set)
        hbar.pack(side="bottom",fill="x")
        vbar.pack(side="right",fill="y")
        self.canvas.pack(side="left",fill="both",expand=True)
        self.canvas.bind("<Configure>",lambda e:self._draw_scaled())

    def _refresh_map(self,path=None,src=None,dst=None):
        self._rendered=render_map(self.base_img,path,src,dst)
        self._draw_scaled()

    def _draw_scaled(self):
        if not hasattr(self,"_rendered"): return
        cw=self.canvas.winfo_width(); ch=self.canvas.winfo_height()
        if cw<10 or ch<10: return
        iw,ih=self._rendered.width,self._rendered.height
        scale=min(cw/iw,ch/ih)
        nw,nh=max(1,int(iw*scale)),max(1,int(ih*scale))
        scaled=self._rendered.resize((nw,nh),Image.LANCZOS)
        self.tk_img=ImageTk.PhotoImage(scaled)
        self.canvas.config(scrollregion=(0,0,nw,nh))
        self.canvas.delete("all")
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_img)

    def find_path(self):
        src=self.src_var.get(); dst=self.dst_var.get()
        if not src or not dst:
            messagebox.showwarning("Missing","Select both source and destination."); return
        if src==dst:
            messagebox.showinfo("Same","Same location!"); return
        self.status.config(text="⏳ Finding path...",fg="#f9e2af"); self.root.update()
        path=astar(self.walkable,BUILDINGS[src],BUILDINGS[dst])
        if not path:
            self.status.config(text="❌ No path found.",fg="#f38ba8")
            messagebox.showerror("No Path","Could not find a path."); return
        dist=sum(math.hypot(path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]) for i in range(len(path)-1))
        dist_m=dist*0.7; walk_min=dist_m/80
        self.status.config(text=f"✅  {src}  →  {dst}   |   ~{dist_m:.0f} m   |   ~{walk_min:.1f} min walk",fg="#a6e3a1")
        self._refresh_map(path,src,dst)

    def clear_path(self):
        self.src_var.set(""); self.dst_var.set("")
        self.status.config(text="Select source and destination.",fg="#a6e3a1")
        self._refresh_map()

if __name__=="__main__":
    try:
        root=tk.Tk()
        CampusMapApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
