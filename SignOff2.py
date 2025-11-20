# app.py
# Infinity Drain – Flanged Channel Configurator (Top + Front views, up to 3 outlets)
# - A (inside length) 12"–96" decimal (4 dp)
# - Inside width fixed 2-1/2"
# - Flange default 1.00" (FCS can toggle 1/2")
# - Outlets measured FROM INNER-RIGHT to center (decimal)
# - Clearance: outlet ≥ 2.00" from BOTH inner ends
# - Sign-Off (Option B): overlay our drawn preview + filled fields onto the official PDF template

import io
import re
import requests
from datetime import date
from dataclasses import dataclass
from typing import List
import math
import streamlit as st
from streamlit.components.v1 import html
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib import colors
from PyPDF2 import PdfReader, PdfWriter

st.set_page_config(
    page_title="Custom Configurator",
    page_icon="data:image/webp;base64,UklGRsQBAABXRUJQVlA4ILgBAABwCwCdASpeAEEAPpU8mEglo6KhOZK4ALASiWknAAH1JnmL9J8AO4GL6aQ2miTG/KUa/r8UXmoN7tYlkK5c3GcGPT7TCNFnNdBsILHcYm3BkleZOlLmRz4vVOpSXVmniD//mWfAAP7+Ez/7P/imicc935CwIgCZjjtdS+Yj+uQbCd+46ta9+Pkcq4mQho9uR+47jfnP/xcwWYWlL/H5RVaENfGC+/Y/4qR9KqEWU1Yh215efqxknue2lo5/7JpB9Fz2b3h3D58BpLGvKVuxgsCgJOqht69AuKOBHP6zE80Rql3AR2WY35dysbv+oKphQITrjymuCBtwQO9sX/UnOAo1G8DD7OLHsmj29BvOdKg6rhie2qkwwn/2aafOm1CH220iw+dQi6MeyhJ//0X6dKSxpOtuKOq3MdRo6LFEDrjWevODNE8fTI+jgc+qlPBl8/HbhVp4LNse9tsGe9I+ehHF+sqqGOlO7duiwejalxNGBu76PUwUP98+CpijLWs9oZphl0y1n/SNBOecqxx3ZEgeWsS+F57XP/1PS//KBstAINtpV2/Wrntx38Daf9GvB2jm7e8evQAAAA==",
    layout="centered",
)

st.markdown("""
    <style>
        /* Tighten top & bottom padding of main content */
        div.block-container {
            padding-top: 0.05rem !important;   /* small gap under the logo */
            padding-bottom: 0.25rem !important;
        }

        /* Optional: slightly tighten sidebar padding too */
        section[data-testid="stSidebar"] > div {
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
        }
    </style>
""", unsafe_allow_html=True)



# -----------------------------
# Branding
# -----------------------------
st.logo('https://uploads-ssl.webflow.com/61dd080b3d2f9fec43b08948/61e5d398a2373f0282831465_InfinityDrainLogo_blk.png')
st.write("")



# -----------------------------
# Constants
# -----------------------------
CHANNEL_WIDTH_IN = 2.50
FLANGE_IN = 1.00
CORNER_RADIUS_IN = 0.50
OUTLET_NOMINAL_DIA_IN = 2.00
MIN_A = 12.0
MAX_A = 96.0

PAGE_SIZE = landscape(LETTER)  # (792, 612)
PAGE_W, PAGE_H = PAGE_SIZE

# Front-view outlet drop (C) by model
OUTLET_DROP = {
    "FX": 2.0,
    "FF": 3.0,
    "FCS": 3.0,
    "FCB": 2.0,
}

MODELS = {
    "FX":  "FX - Fixed Length",
    "FF":  "FF - Fixed Flange",
    "FCS": "FCS - Flange with Schluter",
    "FCB": "FCB - Double Waterproofing",
}

# Official sign-off template
TEMPLATE_URL = "https://infinitydrain-my.sharepoint.com/:b:/p/jjames/EeTQTPMsT8JBvStzsGr1qDQBY26rrl47sMzXYfiI7seWXw?e=W7r4hx&download=1"

# Bottom fields A / B1 / B2 / B3 (your calibrated coords)
A_X,  A_Y  = 95, 427
B1_X, B1_Y = 195, 427
B2_X, B2_Y = 295, 427
B3_X, B3_Y = 395, 427

# Top title row (MODEL / GENERATED ON)
GEN_X,   GEN_Y   = 715, 140    # under "GENERATED ON"

# "Channel" field in the details row
CHANNEL_X, CHANNEL_Y = 105, 75

# ----- Blank template (local PDF) placement (points from bottom-left) -----
BLANK_X = 100                      # left edge of drawing block
BLANK_Y = 230                     # bottom of drawing block
BLANK_W = 510                     # width of drawing block
BLANK_H = 310                     # height of drawing block

# Flip only the drawing block vertically (to match template orientation)
FLIP_OVERLAY_VERTICAL = True

st.markdown("""
<style>
/* widen center column and add left/right breathing room */
.block-container{max-width: 1600px; padding-left: 2.5rem; padding-right: 2.5rem;}
/* a little space between Streamlit columns */
[data-testid="column"] { padding: 0 1rem; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Unified Drawing Geometry
# -----------------------------
@dataclass
class DrawingGeometry:
    """All computed geometry in inches - renderer-agnostic"""
    A_in: float
    flange_in: float
    channel_width_in: float = 2.50
    corner_radius_in: float = 0.50
    outlet_dia_in: float = 2.00
    model_code: str = "FF"
    outlet_drop_in: float = 3.0
    
    @property
    def outer_length_in(self) -> float:
        return self.A_in + 2 * self.flange_in
    
    @property
    def outer_width_in(self) -> float:
        return self.channel_width_in + 2 * self.flange_in
    
    def outlet_positions_from_left(self, outlets_from_right: List[float]) -> List[float]:
        """Convert right-measured to left-measured"""
        return [self.A_in - o for o in outlets_from_right]
    
    def get_layout(self, outlets_from_right: List[float]) -> dict:
        """Complete layout specification"""
        return {
            'outer_length': self.outer_length_in,
            'outer_width': self.outer_width_in,
            'inner_length': self.A_in,
            'inner_width': self.channel_width_in,
            'flange': self.flange_in,
            'corner_radius': self.corner_radius_in,
            'outlet_dia': self.outlet_dia_in,
            'outlet_drop': self.outlet_drop_in,
            'outlet_positions_left': self.outlet_positions_from_left(outlets_from_right),
            'outlets_from_right': outlets_from_right,
        }


# -----------------------------
# Helpers
# -----------------------------
def dec_label(v: float) -> str:
    """
    Format inches as a mixed fraction in 1/16\" steps, e.g.
      12.0000  ->  12"
      12.0625  ->  12-1/16"
      0.5000   ->  1/2"
      3.7500   ->  3-3/4"
    Assumes values are already snapped to 1/16 (which your app enforces).
    """
    if v is None:
        return ""

    v = float(v)
    sign = "-" if v < 0 else ""
    v = abs(v)

    # convert to 16ths, guard against tiny float noise
    n_16 = round(v * 16 + 1e-6)
    whole = n_16 // 16
    num = n_16 % 16

    if num == 0:
        # pure whole number
        return f'{sign}{int(whole)}"'

    # reduce the fraction (so 8/16 -> 1/2, 4/16 -> 1/4, etc.)
    g = math.gcd(num, 16)
    num //= g
    den = 16 // g

    if whole == 0:
        # just the fraction part (e.g. 1/2")
        return f'{sign}{num}/{den}"'
    else:
        # mixed number (e.g. 12-1/16")
        return f'{sign}{int(whole)}-{num}/{den}"'

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_template_pdf_bytes(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=20, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok and r.content and b"%PDF" in r.content[:8]:
            return r.content
    except Exception:
        pass
    return None

def get_pdf_scale_factors(A_in: float, ref_len: float = 36.0):
    """
    Return (size_scale, pos_scale) for PDF dimensions based on inside length.
    
    - Short drains (≈12") → slightly larger text, closer dims.
    - Ref drain (36")     → baseline.
    - Long drains (72–96) → slightly smaller text, pushed-out dims
      so they don’t collide with the drawing.
    """
    # How long vs a 36" reference
    ratio = A_in / ref_len if ref_len > 0 else 1.0

    # Clamp ratio to avoid crazy scaling
    ratio = max(0.5, min(ratio, 3.0))  # 0.5x–3x of ref length

    # Text size: inverse sqrt so changes are smooth
    #  - 12" (~0.5 ratio)  -> ~1.4x bigger text
    #  - 36" (1.0 ratio)   -> 1.0x
    #  - 96" (~2.67 ratio) -> ~0.6x
    size_scale = 1.0 / math.sqrt(ratio)

    # Position scale: longer drains push dimensions further out,
    # but not more than ~1.5x the baseline.
    pos_scale = min(1.5, ratio)

    return size_scale, pos_scale

def get_arrow_scale(A_in: float,
                    min_len: float = 12.0,
                    max_len: float = 96.0,
                    s_short: float = 0.8,   # scale at 12"
                    s_long: float = 4):  # scale at 96"
    """
    Returns a factor so arrowheads are a bit smaller on short drains
    and a bit bigger on long drains.
    12" → ~0.8x, 96" → ~1.3x (tweak as you like).
    """
    t = (A_in - min_len) / (max_len - min_len)
    t = max(0.0, min(1.0, t))  # clamp 0–1
    return s_short + (s_long - s_short) * t

def get_view_spacing(A_in: float) -> float:
    """
    Dynamic vertical spacing between top and front views.
    Returns a value in PDF points.
    Short drains → small spacing
    Long drains  → larger spacing
    """

    # define ranges
    min_len, max_len = 12.0, 96.0

    # spacing behavior: 20 pts at 12", 110 pts at 96"
    spacing_short = 60
    spacing_long  = 130

    t = (A_in - min_len) / (max_len - min_len)
    t = max(0.0, min(1.0, t))   # clamp between 0 and 1

    return spacing_short + (spacing_long - spacing_short) * t


# -----------------------------
# Unified ReportLab Drawing Functions
def draw_top_view_reportlab(
    c,
    geom: DrawingGeometry,
    outlets_from_right: List[float],
    x: float,
    y: float,
    w: float,
    h: float,
    ppi_override: float | None = None
):
    """Draw top view with ALL labels matching SVG, dimension stack scales with length."""
    layout = geom.get_layout(outlets_from_right)

    # Use INSIDE length (A) for scaling, so 12" and 36" share the same baseline look.
    length_for_scale = geom.A_in

    # --- Unified scaling for PDF ---
    size_scale, pos_scale = get_pdf_scale_factors(length_for_scale)

    # Scale to fit drawing block
    if ppi_override is not None:
        ppi = ppi_override
    else:
        ppi = min(
            w / layout['outer_length'],
            h / (layout['outer_width'] + 1.5),
        )

    def P(inches: float) -> float:
        return inches * ppi

    # Center in available space – shifted down a bit to leave room for A-dim
    ox = x + (w - P(layout['outer_length'])) / 2.0
    oy = y + (h - P(layout['outer_width'] + 1.5)) / 2.0 + P(1.0)

    ix = ox + P(geom.flange_in)
    iy = oy + P(geom.flange_in)

    # === CHANNEL & OUTLETS ===
    c.setStrokeColor(colors.black)

    if geom.model_code == "FX" or geom.flange_in == 0.0:
        # FX: fixed-length channel, no flange, no radius
        c.setLineWidth(1.5)
        c.rect(
            ox, oy,
            P(layout['outer_length']),
            P(layout['outer_width']),
            stroke=1, fill=0
        )

        # FX hem in TOP VIEW: two strips along the long sides
        hem_w_in = 0.125
        hem_h = P(hem_w_in)

        c.setLineWidth(1)

        # Top hem strip
        c.rect(
            ox,
            oy,
            P(layout['outer_length']),
            hem_h,
            stroke=1,
            fill=0
        )

        # Bottom hem strip
        c.rect(
            ox,
            oy + P(layout['outer_width']) - hem_h,
            P(layout['outer_length']),
            hem_h,
            stroke=1,
            fill=0
        )

    else:
        # Flanged models: outer rounded, inner channel
        c.setLineWidth(1.5)
        c.roundRect(
            ox, oy,
            P(layout['outer_length']),
            P(layout['outer_width']),
            P(geom.corner_radius_in),
            stroke=1, fill=0
        )

        c.setLineWidth(1)
        c.rect(
            ix, iy,
            P(layout['inner_length']),
            P(layout['inner_width']),
            stroke=1, fill=0
        )

    # Outlet circles – same for all models
    cy = iy + P(layout['inner_width'] / 2.0)
    orad = P(geom.outlet_dia_in / 2.0)
    for x_left in layout['outlet_positions_left']:
        cx = ix + P(x_left)
        c.circle(cx, cy, orad, stroke=1, fill=0)

    # === A DIMENSION (dynamically scaled) ===
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)

    base_offset_in = 0.80           # vertical distance from channel at baseline
    offset_in      = base_offset_in * pos_scale
    dim_y          = oy - P(offset_in)

    base_ah_in = 0.22  # baseline arrowhead size in inches (tweakable)
    arrow_scale = get_arrow_scale(length_for_scale)
    ah = P(base_ah_in * arrow_scale)

    c.line(ix, dim_y, ix + P(layout['inner_length']), dim_y)
    # left arrowhead
    c.line(ix, dim_y, ix + ah, dim_y + ah * 0.4)
    c.line(ix, dim_y, ix + ah, dim_y - ah * 0.4)

    # right arrowhead
    right = ix + P(layout['inner_length'])
    c.line(right, dim_y, right - ah, dim_y + ah * 0.4)
    c.line(right, dim_y, right - ah, dim_y - ah * 0.4)

    # Text size scales (short → bigger, long → smaller) with clamps
    base_font_A = 12.0
    font_A = max(8, min(14, base_font_A * size_scale))

    base_text_offset_in = 0.20
    text_offset_in      = base_text_offset_in * pos_scale

    c.saveState()
    text_cx = ix + P(layout['inner_length'] / 2.0)
    text_cy = dim_y - P(text_offset_in)
    if FLIP_OVERLAY_VERTICAL:
        c.translate(text_cx, text_cy)
        c.scale(1, -1)
        c.setFont("Helvetica-Bold", font_A)
        c.drawCentredString(0, 3, dec_label(geom.A_in))
    else:
        c.setFont("Helvetica-Bold", font_A)
        c.drawCentredString(text_cx, text_cy, dec_label(geom.A_in))
    c.restoreState()


    # (Inside-width and flange notes are still optional / commented out)



def draw_front_view_reportlab(
    c,
    geom: DrawingGeometry,
    outlets_from_right: List[float],
    x: float,
    y: float,
    w: float,
    h: float,
    ppi_override: float | None = None
):
    """Draw front view with B dimensions that scale with overall length."""
    layout = geom.get_layout(outlets_from_right)

    # Use A (inside length) to decide how much to scale dimensions
    length_for_scale = geom.A_in

    # --- Unified scaling for PDF ---
    size_scale, pos_scale = get_pdf_scale_factors(length_for_scale)

    # --- vertical offset for flipped B-label text (points) ---
    # For a 12" drain we like ~7, for a 96" drain ~3.
    min_len, max_len = 12.0, 96.0
    y_short, y_long = 7.0, 3.0  # 12" -> 7, 96" -> 3

    t = (length_for_scale - min_len) / (max_len - min_len)
    t = max(0.0, min(1.0, t))   # clamp 0–1
    flip_y_offset = y_short + (y_long - y_short) * t

    tray_thick_in = 0.45
    drop_in = layout['outlet_drop']
    total_h_in = tray_thick_in + drop_in + 2.0

    # Fit into block
    if ppi_override is not None:
        ppi = ppi_override
    else:
        ppi = min(w / layout['outer_length'], h / total_h_in)

    def P(inches: float) -> float:
        return inches * ppi

    ox = x + (w - P(layout['outer_length'])) / 2.0
    base_y = y + P(1.5)   # visual baseline

    # === CHANNEL & OUTLETS ===
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.line(ox, base_y, ox + P(layout['outer_length']), base_y)

    c.setLineWidth(1)

    # For FX, no flange → no inset. Others keep the current 1" inset look.
    if geom.model_code == "FX" or geom.flange_in == 0.0:
        inset = 0.0
    else:
        inset = P(1.0)

    tray_x = ox + inset
    tray_w = P(layout['outer_length']) - 2 * inset
    c.rect(tray_x, base_y, tray_w, P(tray_thick_in), stroke=1, fill=0)

    # FX hem in FRONT VIEW: strip that goes down from the top
    if geom.model_code == "FX" or geom.flange_in == 0.0:
        hem_drop_in = 0.125 
        hem_h = P(hem_drop_in)

        # Top of tray in this flipped coordinate system
        tray_top_y = base_y  # rect starts at base_y and goes down

        c.setLineWidth(1)
        c.rect(
            tray_x,
            tray_top_y,
            tray_w,
            hem_h,
            stroke=1,
            fill=0
        )


    iv_left = ox + P(geom.flange_in)
    iv_right = iv_left + P(layout['inner_length'])

    drop_w = P(geom.outlet_dia_in * 0.9)

    for i, (x_left, b_val) in enumerate(zip(layout['outlet_positions_left'],
                                            outlets_from_right), start=1):
        cx = ox + P(geom.flange_in) + P(x_left)

        # --- outlet body (always solid black) ---
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(
            cx - drop_w/2,
            base_y + P(tray_thick_in),
            drop_w,
            P(drop_in),
            stroke=1, fill=0
        )

        # --- CL line (light grey) ---
        c.setStrokeColor(colors.Color(0, 0, 0, alpha=0.35))
        c.setLineWidth(0.8)
        c.line(
            cx, base_y - P(0.15),
            cx, base_y + P(tray_thick_in + drop_in) + P(0.15)
        )

        # CL label
        base_font_cl = 10.0
        font_cl = max(7, min(12, base_font_cl * size_scale))

        c.saveState()
        cl_y = base_y - P(0.30)
        if FLIP_OVERLAY_VERTICAL:
            c.translate(cx, cl_y)
            c.scale(1, -1)
            c.setFont("Helvetica", font_cl)
            c.drawCentredString(0, 3, "CL")
        else:
            c.setFont("Helvetica", font_cl)
            c.drawCentredString(cx, cl_y, "CL")
        c.restoreState()

    # === B DIMENSION STACK (above drawing, scaled) ===
    base_dim_offset_in = 1.4   # distance above outlet drop at baseline
    base_gap_in        = 1.4   # gap between B1/B2/B3 at baseline

    # arrowhead: smaller on short, bigger on long
    base_ah_in   = 0.22
    arrow_scale  = get_arrow_scale(length_for_scale)
    ah           = P(base_ah_in * arrow_scale)

    dim_offset_in = base_dim_offset_in * pos_scale
    gap_in        = base_gap_in * pos_scale

    base_dim_y = base_y + P(tray_thick_in + drop_in + dim_offset_in)
    gap = P(gap_in)

    c.setLineWidth(1)

    for i, (x_left, b_val) in enumerate(zip(layout['outlet_positions_left'],
                                            outlets_from_right), start=1):
        cx = ox + P(geom.flange_in) + P(x_left)
        dim_y = base_dim_y + (i - 1) * gap

        # ensure every B-dim uses solid black for lines + text
        c.setStrokeColor(colors.black)
        c.setFillColor(colors.black)

        # dimension line
        c.line(cx, dim_y, iv_right, dim_y)
        # right arrowhead
        c.line(iv_right, dim_y, iv_right - ah, dim_y + ah * 0.4)
        c.line(iv_right, dim_y, iv_right - ah, dim_y - ah * 0.4)

        # left arrowhead
        c.line(cx, dim_y, cx + ah, dim_y + ah * 0.4)
        c.line(cx, dim_y, cx + ah, dim_y - ah * 0.4)

        # dimension label
        base_font_B = 12.0
        font_B = max(8, min(14, base_font_B * size_scale))

        label_x = (cx + iv_right) / 2.0
        label_y = dim_y + P(0.15)

        c.saveState()
        if FLIP_OVERLAY_VERTICAL:
            c.translate(label_x, label_y)
            c.scale(1, -1)
            c.setFont("Helvetica-Bold", font_B)
            c.drawCentredString(0, flip_y_offset, dec_label(b_val))
        else:
            c.setFont("Helvetica-Bold", font_B)
            c.drawCentredString(label_x, label_y, dec_label(b_val))
        c.restoreState()



def draw_combined_preview_reportlab(
    c,
    geom: DrawingGeometry,
    outlets_from_right: List[float],
    x: float,
    y: float,
    w: float,
    h: float
):
    """Draw both views stacked vertically with shared horizontal scale."""
    # Split the available height between front & top
    h_top = h * 0.45
    h_front = h * 0.55

    # Dynamic vertical spacing between the two views
    spacing = get_view_spacing(geom.A_in)

    # --- Compute a SINGLE ppi so top & front have the same width ---
    outer_len_in = geom.outer_length_in
    v_top_in = geom.outer_width_in + 1.5          # same as top-view logic
    tray_thick_in = 0.45
    v_front_in = tray_thick_in + geom.outlet_drop_in + 2.0  # same as front-view logic

    # Guard against divide-by-zero
    candidates = []
    if outer_len_in > 0:
        candidates.append(w / outer_len_in)
    if v_top_in > 0:
        candidates.append(h_front / v_top_in)
    if v_front_in > 0:
        candidates.append(h_top / v_front_in)

    ppi = min(candidates) if candidates else 1.0

    # --- Draw top view (uses shared ppi for x-scale) ---
    draw_top_view_reportlab(
        c,
        geom,
        outlets_from_right,
        x,
        y,
        w,
        h_front,
        ppi_override=ppi
    )

    # --- Draw front view with same horizontal scale (ppi) ---
    draw_front_view_reportlab(
        c,
        geom,
        outlets_from_right,
        x,
        y + h_front + spacing,
        w,
        h_top,
        ppi_override=ppi
    )

    # View labels - counter-flip text
    c.saveState()
    if FLIP_OVERLAY_VERTICAL:
        # TOP VIEW label
        label_x = x + 5
        label_y = y + h_front - 150
        c.translate(label_x, label_y)
        c.scale(1, -1)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(-50, 30, "TOP VIEW")
        c.restoreState()

        # FRONT VIEW label
        c.saveState()
        label_x = x + 5
        label_y = y + h - 75
        c.translate(label_x, label_y)
        c.scale(1, -1)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(-50, 10, "FRONT VIEW")
        c.restoreState()
    else:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x-10, y + h, "TOP VIEW")
        c.drawString(x-10, y + h_front - 15, "FRONT VIEW")
        c.restoreState()


def max_outlets_for_length(A_in: float) -> int:
    """
    12–<48  -> 1 outlet
    48–<72  -> up to 2 outlets
    72–96   -> up to 3 outlets
    Always returns at least 1 so the slider never has an empty options list.
    """
    if A_in < 24.0:
        return 2
    elif A_in < 36.0:
        return 2
    else:
        return 3

def draw_title_block_text(
    c,
    model_code: str,  # FF / FCS / FCB / NDC FCS (already “final” code)
    geom: DrawingGeometry,
    outlets_from_right_in: list[float],
):
    """
    Writes channel (model code only), A length, B1/B2/B3,
    and date into the template title block.
    """

    # 1) CHANNEL – model code only
    c.setFont("Helvetica", 10)
    c.drawString(CHANNEL_X, CHANNEL_Y, model_code)

    # 2) Date (GENERATED ON field)
    today_str = date.today().strftime("%m/%d/%Y")
    c.setFont("Helvetica", 9)
    c.drawString(GEN_X, GEN_Y, today_str)

    # 3) Length (A)
    c.drawString(275, 35, dec_label(geom.A_in))

    # 4) Outlet locations
    if len(outlets_from_right_in) >= 1:
        c.drawString(435, 80, dec_label(outlets_from_right_in[0]))
    if len(outlets_from_right_in) >= 2:
        c.drawString(435, 55, dec_label(outlets_from_right_in[1]))
    if len(outlets_from_right_in) >= 3:
        c.drawString(435, 30, dec_label(outlets_from_right_in[2]))


# -----------------------------
# PDF Generation
# -----------------------------

def build_blank_template_overlay(
    model_code: str,
    A_in: float,
    outlets_from_right_in: list[float],
    flange_in: float,
) -> bytes:
    """Generate PDF overlay using unified drawing functions"""
    geom = DrawingGeometry(
        A_in=A_in,
        flange_in=flange_in,
        model_code=model_code,
        outlet_drop_in=OUTLET_DROP.get(model_code, 3.0)
    )
    
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=PAGE_SIZE)

    
    c.saveState()
    if FLIP_OVERLAY_VERTICAL:
        cx = BLANK_X + BLANK_W / 2.0
        cy = BLANK_Y + BLANK_H / 2.0
        c.translate(cx, cy)
        c.scale(1, -1)
        dx, dy, dw, dh = -BLANK_W/2.0, -BLANK_H/2.0, BLANK_W, BLANK_H
    else:
        dx, dy, dw, dh = BLANK_X, BLANK_Y, BLANK_W, BLANK_H
    
    # 1) Draw the flipped drawing block
    draw_combined_preview_reportlab(c, geom, outlets_from_right_in, dx, dy, dw, dh)
    c.restoreState()  # back to normal (non-flipped) page coords

    # 2) Stamp title-block info (Channel, Model, Length, Outlets, Date)
    draw_title_block_text(
        c,
        model_code=model_code,
        geom=geom,
        outlets_from_right_in=outlets_from_right_in,
    )

    # 3) Finish the overlay page
    c.save()

    buf.seek(0)
    return buf.getvalue()


def merge_overlay_on_template_bytes(template_pdf_bytes: bytes, overlay_bytes: bytes) -> bytes:
    base_reader = PdfReader(io.BytesIO(template_pdf_bytes))
    overlay_reader = PdfReader(io.BytesIO(overlay_bytes))
    page = base_reader.pages[0]
    page.merge_page(overlay_reader.pages[0])
    writer = PdfWriter(); writer.add_page(page)
    out = io.BytesIO(); writer.write(out); out.seek(0)
    return out.getvalue()
def snap_to_sixteenth(value: float | None) -> float | None:
    """Snap a value down to the nearest 1/16\" increment (always rounding down)."""
    if value is None:
        return None
    snapped = math.floor(float(value) * 16.0) / 16.0
    return round(snapped, 4)

def on_change_A_in():
    """Snap A_in to 1/16 and then reseed outlets."""
    val = st.session_state.get("A_in")
    if isinstance(val, (int, float)):
        st.session_state["A_in"] = snap_to_sixteenth(val)
    reseed_outlets()

def snap_outlet(i: int):
    """Snap outlet Bi to 1/16 when user leaves the field."""
    key = f"b_{i}"
    val = st.session_state.get(key)
    if isinstance(val, (int, float)):
        st.session_state[key] = snap_to_sixteenth(val)

# --- callbacks: reseed outlets when A or count changes ---
def reseed_outlets():
    A = st.session_state["A_in"]
    n = st.session_state["count"]
    min_right = 2.0
    max_right = max(min_right, A - 2.0)

    # equidistant from INSIDE-RIGHT: A/(n+1), 2A/(n+1), ...
    step = A / (n + 1)
    for i in range(1, n + 1):
        v = step * i
        # clamp to allowed range
        v = max(min_right, min(v, max_right))
        # snap to nearest 1/16 down
        v = snap_to_sixteenth(v)
        # make sure snapping didn't push us below min_right
        if v < min_right:
            v = min_right
        st.session_state[f"b_{i}"] = v


    for i in range(n + 1, 4):
        st.session_state.pop(f"b_{i}", None)


# -----------------------------
# On-screen SVG renderer with styled containers
# -----------------------------
def make_top_view_svg(A_in, flange_in, outlets_from_right_in, px_per_in=32):
    """Generate TOP VIEW SVG with dimensions"""
    f = flange_in
    outer_L_in = A_in + 2*f
    outer_W_in = CHANNEL_WIDTH_IN + 2*f

    # FX / any no-flange model: treat as plain channel, no radius, no inner rect
    is_no_flange = (f == 0.0)

    # --- DIMENSION SCALING ---
    ref_len = 24.0
    max_dim_scale = 5.0
    dim_scale = max(1.0, min(outer_L_in / ref_len, max_dim_scale))

    size_scale = dim_scale          # text/arrow size
    pos_scale  = min(dim_scale, 1.5)  # how far we move them (clamped)

    margin_px = max(60, int(px_per_in * 0.6))
    Lpx_out = outer_L_in * px_per_in
    Wpx_out = outer_W_in * px_per_in
    Lpx_in  = A_in * px_per_in
    Wpx_in  = CHANNEL_WIDTH_IN * px_per_in

    # radius only when we actually have a flange
    rpx = 0 if is_no_flange else CORNER_RADIUS_IN * px_per_in

    ix = margin_px + f * px_per_in
    iy = margin_px + f * px_per_in

    cx_list = []
    for o_right in outlets_from_right_in:
        x_from_left_in = A_in - o_right
        cx_list.append(ix + x_from_left_in * px_per_in)

    cy = iy + Wpx_in/2.0
    orad = (OUTLET_NOMINAL_DIA_IN * 0.9 / 2.0) * px_per_in

    total_w = int(Lpx_out + margin_px*2)
    total_h = int(Wpx_out + margin_px*2)
    pad_top    = 0
    pad_bottom = 0.0

    def line(x1,y1,x2,y2,stroke="#000",width=1):
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"/>'

    svg = [(
        f'<svg viewBox="0 {-pad_top:.2f} {total_w:.2f} {total_h + pad_top + pad_bottom:.2f}" '
        f'width="100%" preserveAspectRatio="xMinYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" style="overflow:visible">'
    )]

    # Outer shape: for FX (no flange) this is the channel; for others it's outer flange
    svg.append(
        f'<rect x="{margin_px}" y="{margin_px}" '
        f'width="{Lpx_out}" height="{Wpx_out}" '
        f'rx="{rpx}" ry="{rpx}" fill="none" stroke="#000" stroke-width="1.4"/>'
    )

    # Inner channel only when there’s a flange
    if not is_no_flange:
        svg.append(
            f'<rect x="{ix}" y="{iy}" width="{Lpx_in}" height="{Wpx_in}" '
            f'fill="none" stroke="#000" stroke-width="1"/>'
        )

    # FX hem in TOP VIEW: two strips along the long sides
    if is_no_flange:  # FX
        hem_w_in = 0.125  
        hem_h_px = hem_w_in * px_per_in

        # Top hem strip
        svg.append(
            f'<rect x="{margin_px}" y="{margin_px}" '
            f'width="{Lpx_out}" height="{hem_h_px:.2f}" '
            f'fill="none" stroke="#000" stroke-width="1"/>'
        )

        # Bottom hem strip
        svg.append(
            f'<rect x="{margin_px}" y="{margin_px + Wpx_out - hem_h_px:.2f}" '
            f'width="{Lpx_out}" height="{hem_h_px:.2f}" '
            f'fill="none" stroke="#000" stroke-width="1"/>'
        )


    for cx in cx_list:
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{orad}" '
            f'fill="none" stroke="#000" stroke-width="1"/>'
        )

    # A dimension (scaled)
    base_ah = 7.0
    ah = base_ah * size_scale

    base_dy_offset = 30.0
    dy = margin_px - base_dy_offset * pos_scale
    dy = max(20, dy)

    svg.append(line(ix, dy, ix + Lpx_in, dy, width=1))
    svg.append(line(ix, dy, ix + ah, dy - ah/2, width=1))
    svg.append(line(ix, dy, ix + ah, dy + ah/2, width=1))
    svg.append(line(ix + Lpx_in, dy, ix + Lpx_in - ah, dy - ah/2, width=1))
    svg.append(line(ix + Lpx_in, dy, ix + Lpx_in - ah, dy + ah/2, width=1))

    base_font_A = 12.0
    font_A = base_font_A * size_scale

    base_text_offset = 8.0
    text_offset = base_text_offset * size_scale

    svg.append(
        f'<text x="{ix + Lpx_in/2:.2f}" y="{dy - text_offset:.2f}" '
        f'text-anchor="middle" font-family="Helvetica" font-weight="700" '
        f'font-size="{font_A:.1f}">{dec_label(A_in)}</text>'
    )

    svg.append('</svg>')
    return "\n".join(svg)



def make_front_view_svg(A_in, flange_in, outlets_from_right_in, px_per_in=32, model_code="FF"):
    f = flange_in
    outer_L_in = A_in + 2*f
    Lpx_out = outer_L_in * px_per_in
    margin_px = max(60, int(px_per_in * 0.6))
    vertical_offset = -100  # raise front view to reveal dimension lines

    # --- DIMENSION SCALING ---
    ref_len = 24.0
    max_dim_scale = 5.0
    dim_scale = max(1.0, min(outer_L_in / ref_len, max_dim_scale))

    size_scale = dim_scale
    pos_scale  = min(dim_scale, 3.0)

    total_w = int(Lpx_out + margin_px*2)
    total_h = int(margin_px*2 +140)  # much taller
    pad_top    = 0.0
    pad_bottom = 0.0  # extra footroom for stacked B1/B2/B3 dims

    iv_left  = margin_px + f * px_per_in
    iv_right = iv_left + A_in * px_per_in
    drop_px  = OUTLET_DROP.get(model_code, 3.0) * px_per_in
    tray_th  = 0.45 * px_per_in

    def line(x1,y1,x2,y2,stroke="#000",width=1):
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"/>'

    svg = [(
        f'<svg viewBox="0 0 {total_w:.2f} {total_h + pad_top + pad_bottom:.2f}" '
        f'width="100%" preserveAspectRatio="xMinYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" style="overflow:visible">'
    )]
    # svg.append(
    # f'<rect x="0" y="0" width="{total_w:.2f}" height="{total_h + pad_top + pad_bottom:.2f}" '
    # f'fill="none" stroke="red" stroke-width="1" stroke-dasharray="6,4" />'
    # )
    svg.append(line(margin_px, margin_px + vertical_offset + 100, margin_px + Lpx_out, margin_px + vertical_offset + 100, width=1.4))
    svg.append(
        f'<rect x="{iv_left:.2f}" '
        f'y="{margin_px + vertical_offset + 100:.2f}" '
        f'width="{(A_in * px_per_in):.2f}" '
        f'height="{tray_th:.2f}" '
        f'fill="none" stroke="#000" stroke-width="1"/>'
    )

        # FX hem in FRONT VIEW SVG: strip that goes down from the top
    if model_code == "FX":
        hem_drop_in = 0.125
        hem_h_px = hem_drop_in * px_per_in
        lip_y = margin_px + vertical_offset + 100  # top of tray

        svg.append(
            f'<rect x="{iv_left:.2f}" '
            f'y="{lip_y:.2f}" '
            f'width="{(A_in * px_per_in):.2f}" '
            f'height="{hem_h_px:.2f}" '
            f'fill="none" stroke="#000" stroke-width="1"/>'
        )


    # outlets
    cx_list = []
    for o_right in outlets_from_right_in:
        cx_list.append(iv_left + (A_in - o_right) * px_per_in)
    for cx in cx_list:
        svg.append(f'<rect x="{cx - 18:.2f}" y="{margin_px + vertical_offset + 114.5:.2f}" width="36" height="{drop_px - 36:.2f}" fill="none" stroke="#000" stroke-width="1"/>')

    # --- Centerlines (CL) for each outlet (scaled) ---
    tray_thick_px = max(2.0, 0.5 * px_per_in)
    lip_y         = margin_px + vertical_offset + 100
    base_cl_up   = 14.0
    base_cl_down = 14.0
    cl_up   = base_cl_up * pos_scale         # how far CL goes above
    cl_down = base_cl_down * pos_scale       # below

    base_font_cl = 11.0
    font_cl = base_font_cl * size_scale      # label size

    for cx in cx_list:
        svg.append(
            f'<line x1="{cx:.2f}" y1="{lip_y - cl_up:.2f}" '
            f'x2="{cx:.2f}" y2="{lip_y + drop_px + cl_down:.2f}" '
            f'stroke="#000" stroke-width="1" stroke-dasharray="6,4" opacity="0.6" />'
        )
        base_font_cl = 11.0
        font_cl = base_font_cl * dim_scale
        svg.append(
            f'<text x="{cx:.2f}" y="{lip_y - (cl_up + 4):.2f}" '
            f'text-anchor="middle" font-family="Helvetica" '
            f'font-size="{font_cl:.1f}" font-weight="700">CL</text>'
        )

    # dimensions (scaled)
    base_y_offset = 5.0
    base_gap      = 25.0
    base_ah       = 7.0

    base_y = (margin_px + vertical_offset + 100 +
            drop_px + base_y_offset * pos_scale)
    gap = base_gap * pos_scale
    ah  = base_ah * size_scale

    base_font_b = 12.0
    font_b = base_font_b * size_scale
    base_text_offset = 4.0
    text_offset = base_text_offset * size_scale


    for i, cx in enumerate(cx_list, start=1):
        y = base_y + (i-1)*gap
        svg.append(line(cx, y, iv_right, y, width=1))
        svg.append(line(iv_right, y, iv_right - ah, y - ah/2, width=1))
        svg.append(line(iv_right, y, iv_right - ah, y + ah/2, width=1))
        svg.append(line(cx, y, cx + ah, y - ah/2, width=1))
        svg.append(line(cx, y, cx + ah, y + ah/2, width=1))
        val = outlets_from_right_in[i-1]

        base_font_b = 12.0
        font_b = base_font_b * dim_scale
        base_text_offset = 4.0
        text_offset = base_text_offset * dim_scale

        svg.append(
            f'<text x="{(cx+iv_right)/2:.2f}" y="{y - text_offset:.2f}" '
            f'text-anchor="middle" font-family="Helvetica" font-weight="700" '
            f'font-size="{font_b:.1f}">{dec_label(val)}</text>'
        )

    svg.append('</svg>')
    return "\n".join(svg)

# -----------------------------
# Landing / title screen state
# -----------------------------
if "started" not in st.session_state:
    st.session_state["started"] = False

# If NOT started → show the landing page and STOP
if not st.session_state["started"]:
    st.markdown("""
        <div style="text-align:center; margin-top:60px;">
            <img src="https://uploads-ssl.webflow.com/61dd080b3d2f9fec43b08948/61e5d398a2373f0282831465_InfinityDrainLogo_blk.png" width="220"/>
            <h2>Custom Configurator</h2>
            <p>Generate shop drawings and sign-offs for FX, FF, FCS, and FCB channels.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("▶️ Begin", use_container_width=True):
        st.session_state["started"] = True
        st.rerun()  # Immediately refresh into configurator mode

    st.stop()  # Do NOT render anything else

# -----------------------------
# UI
# -----------------------------
left_col, right_col = st.columns([7, 5])

with right_col:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.title("Custom Configurator")
    st.divider()
    model_code = st.selectbox("Select Series", list(MODELS.keys()),
                            format_func=lambda k: MODELS[k])

    # Default flange = 1" for flanged models
    flange_in_ui = FLANGE_IN

    if model_code == "FCS":
        # FCS can have 1/2" flange (NDC FCS)
        use_half_flange = st.toggle('1/2" Flange (NDC FCS)', value=False)
        flange_in_ui = 0.50 if use_half_flange else 1.00
    elif model_code == "FX":
        # FX – Fixed Length uses overall length with no flange
        flange_in_ui = 0.00


    A_label = (
        'A – Overall Inside Length (in)'
        if model_code in ("FF", "FCS", "FCB")
        else 'A – Overall Length (in)'  # FX wording
    )

    A_in = st.number_input(
        A_label,
        min_value=MIN_A, max_value=MAX_A, value=36.0000,
        step=0.0625, format="%.4f",
        key="A_in",
        on_change=on_change_A_in,
    )

        # ---- Enforce max outlets based on A_in ----
    max_out = max_outlets_for_length(A_in)

    allowed_counts = list(range(1, max_out + 1))  # e.g. [1], [1,2], or [1,2,3]

    prev_count = st.session_state.get("count", 1)
    if prev_count not in allowed_counts:
        prev_count = allowed_counts[-1]

    default_index = allowed_counts.index(prev_count)
    # st.markdown("""
    # <style>
    # /* Keep the radio title left-aligned (this still works fine) */
    # label[for="count"] {
    #     text-align: left !important;
    #     font-weight: 600;
    # }

    # /* Apply to ALL radiogroups (Streamlit radios) */
    # [role="radiogroup"] {
    #     display: grid !important;
    #     grid-template-columns: repeat(3, 1fr);
    #     width: 100% !important;
    # }

    # /* Base: each radio option is flex so we can align content */
    # [role="radiogroup"] > label {
    #     display: flex !important;
    #     align-items: center !important;
    # }

    # /* 1st option → left slot, left aligned */
    # [role="radiogroup"] > label:nth-of-type(1) {
    #     grid-column: 1;
    #     justify-content: flex-start !important;
    # }

    # /* 2nd option → middle slot, centered (same X whether 2 or 3 options) */
    # [role="radiogroup"] > label:nth-of-type(2) {
    #     grid-column: 2;
    #     justify-content: center !important;
    # }

    # /* 3rd option → right slot, right aligned */
    # [role="radiogroup"] > label:nth-of-type(3) {
    #     grid-column: 3;
    #     justify-content: flex-end !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)



    count = st.radio(
        "Number of outlets",
        options=allowed_counts,
        index=default_index,
        key="count", horizontal=True
    )
    if "last_count" not in st.session_state:
        st.session_state["last_count"] = count

    if st.session_state["last_count"] != count:
        reseed_outlets()
        st.session_state["last_count"] = count

    # # Optional: let the user know why it changed
    # if st.session_state.get("count") != prev_count:
    #     st.warning(
    #         'Outlet count limited by length: 12–<48" → 1 outlet, '
    #         '48–<72" → 2 outlets, 72–96" → 3 outlets.'
    #     )
    
    if ("A_in" in st.session_state) and ("count" in st.session_state) and ("_seeded" not in st.session_state):
        A = st.session_state["A_in"]
        n = st.session_state["count"]
        min_right = 2.0
        max_right = max(min_right, A - 2.0)
        step = A / (n + 1)
        for i in range(1, n + 1):
            v = max(min_right, min(step * i, max_right))
            st.session_state[f"b_{i}"] = round(v, 4)
        st.session_state["_seeded"] = True

    min_right = 2.0
    max_right = max(min_right, st.session_state["A_in"] - 2.0)

    b_cols = st.columns(3)

    def outlet_input(i: int, label: str, disabled: bool):
        if disabled:
            b_cols[i-1].text_input(
                label + " (in)",
                value="",
                key=f"b_{i}_ghost",
                disabled=True,
            )
            return None

        return b_cols[i-1].number_input(
            label + " (in)",
            min_value=min_right,
            max_value=float(max_right),
            value=float(st.session_state.get(f"b_{i}", min(6.0, max_right))),
            step=0.0625,
            format="%.4f",
            key=f"b_{i}",
            on_change=lambda i=i: snap_outlet(i),
        )


    if model_code == "FX":
        b_label_base = "Outlet from Right Outside Edge"
    else:
        b_label_base = "Outlet from Inside Right Edge"

    b1 = outlet_input(1, f"B1 – {b_label_base}", disabled=False)
    b2 = outlet_input(2, f"B2 – {b_label_base}",
                    disabled=(st.session_state["count"] < 2))
    b3 = outlet_input(3, f"B3 – {b_label_base}",
                    disabled=(st.session_state["count"] < 3))

    outlets = []
    for i in range(1, count + 1):
        val = st.session_state.get(f"b_{i}")
        if isinstance(val, (float, int)):
            outlets.append(snap_to_sixteenth(val))

    st.markdown("---")

if outlets and all(isinstance(v, (float, int)) for v in outlets):
    violations = []
    for i, b in enumerate(outlets, start=1):
        if not (2.0 <= b <= A_in - 2.0):
            violations.append(f"B{i} must be between 2.00 and {A_in-2.0:.4f}")

    if violations:
        for v in violations:
            st.error(v)
    else:
        # --- additional validation: uniqueness & min center-to-center spacing ---
        # no duplicates
        if len(set(round(v, 4) for v in outlets)) != len(outlets):
            st.warning("Multiple outlets have the same value. Please check your input.")
        else:
            # --- additional validation: uniqueness & min 3.0" center-to-center spacing ---
            if len(set(round(v, 4) for v in outlets)) != len(outlets):
                st.warning("Multiple outlets have the same value. Please check your input")
            else:
                min_sep = 3.0  # inches
                too_close = []
                for i in range(len(outlets)):
                    for j in range(i + 1, len(outlets)):
                        if abs(outlets[i] - outlets[j]) < min_sep - 1e-6:
                            too_close.append((i + 1, j + 1, abs(outlets[i] - outlets[j])))

                if too_close:
                    st.warning("⚠️ Minimum distance between outlets should be 3\". "
                            "Please check your B dimensions.")
                else:
                    with left_col:
                        st.write("")
                        st.write("")

                        # Generate SVGs with styled containers
                        svg_top = make_top_view_svg(A_in, flange_in_ui, outlets)
                        svg_front = make_front_view_svg(A_in, flange_in_ui, outlets, model_code=model_code)

                        # Fixed-size styled containers - drawing scales inside
                        top_container = f"""
                        <div style="border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:12px;height:320px;display:flex;flex-direction:column;">
                        <div style="font:bold 12px Helvetica, Arial; margin: 2px 0 8px 0;">TOP VIEW</div>
                        <div style="flex:1;display:flex;align-items:center;justify-content:flex-start;">
                        {svg_top}
                        </div>
                        </div>
                        """
                    
                        front_container = f"""
                        <div style="border:1px solid #ddd;border-radius:8px;padding:10px;height:320px;display:flex;flex-direction:column;">
                        <div style="font:bold 12px Helvetica, Arial; margin: 2px 0 8px 0;">FRONT VIEW</div>
                        <div style="flex:1;display:flex;align-items:center;justify-content:flex-start;">
                        {svg_front}
                        </div>
                        </div>
                        """
                        
                        # Display with fixed heights
                        # st.subheader("Top View")
                        html(top_container, height=350, scrolling=False)
                    
                        # st.subheader("Front View")
                        html(front_container, height=350, scrolling=False)

                    tpl_bytes = fetch_template_pdf_bytes(TEMPLATE_URL)
                    if tpl_bytes:
                        # Decide what should actually be printed in the Channel field
                        if model_code == "FCS" and flange_in_ui == 0.50:
                            pdf_model_code = "NDC FCS"
                        else:
                            pdf_model_code = model_code

                        overlay = build_blank_template_overlay(
                            model_code=pdf_model_code,
                            A_in=A_in,
                            outlets_from_right_in=outlets,
                            flange_in=flange_in_ui,
                        )

                        final_pdf = merge_overlay_on_template_bytes(tpl_bytes, overlay)
                        with right_col:
                            st.download_button(
                                "🧾 Download Shop Drawing",
                                data=final_pdf,
                                file_name=f"{model_code} {A_in:.4f} Drawing.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        st.error("Couldn't fetch the blank template. Check TEMPLATE_URL.")

else:
    with left_col:

        st.info("Adjust the options on the right to generate the drawings automatically.")
