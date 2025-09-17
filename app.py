import csv
import io
import math
import os
import random
import string
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from concurrent.futures import TimeoutError as FuturesTimeout

from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid


def _merge_payload():
    # Merge JSON, form en query string. Oorskryf nie JSON sleutels as dit reeds daar is nie.
    data = request.get_json(silent=True) or {}
    for src in (request.form, request.args):
        for k, v in src.items():
            data.setdefault(k, v)
    return data

def _as_int(v, default=0):
    """
    Safely convert a string-like source to an int.
    Falls back to 'default' only if 'default' is numeric, otherwise 0.
    """
    try:
        # Hanteer leë stringe, "none", "null", ens.
        if v is None:
            return int(default)
        s = str(v).strip().lower()
        if s in {"", "none", "null", "nan", "?"}:
            return int(default)
        return int(float(s))
    except Exception:
        try:
            return int(default)
        except (ValueError, TypeError):
            return 0

def _as_float(v, default):
    """
    Draai enige stringagtige bron veilig na float.
    """
    try:
        if v is None:
            raise ValueError("None")
        s = str(v).strip().lower()
        if s in {"", "none", "null", "nan"}:
            raise ValueError("empty")
        return float(s)
    except Exception:
        try:
            return float(default)
        except (ValueError, TypeError):
            return 0.0

def _as_ratio(v, default):
    """
    Laat '15%' of '0.15' of 0.15 toe. Val terug op 'default' of 0.0.
    """
    try:
        if v is None:
            raise ValueError("None")
        s = str(v).strip().lower()
        if s in {"", "none", "null", "nan"}:
            raise ValueError("empty")
        if s.endswith("%"):
            s = s[:-1].strip()
            return float(s) / 100.0
        return float(s)
    except Exception:
        return float(default) if isinstance(default, (int, float)) else 0.0

def _as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory store of uploaded clue set for simplicity
CLUESET: List[Dict[str, str]] = []

# Progress tracking
PROGRESS_STORE: Dict[str, Dict] = {}

# ----------------------------
# Data structures and helpers
# ----------------------------

@dataclass
class Cell:
    ch: Optional[str] = None  # letter if white; None means black/unset
    forbidden: bool = False   # True if this cell is forbidden for word placement

@dataclass
class PlacedWord:
    answer: str
    clue: str
    row: int
    col: int
    horizontal: bool  # True=horiz, False=vert
    length: int

class Grid:
    def __init__(self, rows: int, cols: int, symmetry_type: str = "none", strict_symmetry: bool = False):
        self.rows = rows
        self.cols = cols
        self.symmetry_type = symmetry_type
        self.strict_symmetry = strict_symmetry
        # Start fully black. Cells become white only when a letter is placed.
        self.cells = [[Cell() for _ in range(cols)] for _ in range(rows)]
        self.placed: List[PlacedWord] = []

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_black(self, r: int, c: int) -> bool:
        return self.cells[r][c].ch is None

    def is_forbidden(self, r: int, c: int) -> bool:
        return self.cells[r][c].forbidden

    def letter(self, r: int, c: int) -> Optional[str]:
        return self.cells[r][c].ch

    def set_letter(self, r: int, c: int, ch: str):
        """Set letter without symmetry (for internal use)"""
        self.cells[r][c].ch = ch

    def set_forbidden(self, r: int, c: int):
        """Mark a cell as forbidden for word placement without symmetry"""
        self.cells[r][c].forbidden = True
        self.cells[r][c].ch = None

    def white_count(self) -> int:
        return sum(1 for r in range(self.rows) for c in range(self.cols) if self.cells[r][c].ch is not None)

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get valid neighboring coordinates"""
        neighbors = []
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                neighbors.append((nr, nc))
        return neighbors
    
    # Add this method right after the existing get_neighbors method:
    def get_symmetric_cell(self, r: int, c: int) -> Tuple[int, int]:
        """Get the 180-degree rotationally symmetric cell"""
        return (self.rows - 1 - r, self.cols - 1 - c)
    
    def get_mirror_cell(self, r: int, c: int) -> Tuple[int, int]:
        """Get the vertically mirrored cell (left-right symmetry)"""
        return (r, self.cols - 1 - c)

    def set_letter_with_symmetry(self, r: int, c: int, ch: str):
        """Set letter and maintain symmetry based on symmetry_type"""
        self.cells[r][c].ch = ch
        
        if self.symmetry_type == "rotational":
            sym_r, sym_c = self.get_symmetric_cell(r, c)
            if (sym_r, sym_c) != (r, c):
                self.cells[sym_r][sym_c].ch = ch
        elif self.symmetry_type == "mirror":
            mirror_c = self.cols - 1 - c
            if mirror_c != c:
                self.cells[r][mirror_c].ch = ch

    def set_forbidden_with_symmetry(self, r: int, c: int):
        """Mark cell as forbidden and maintain symmetry"""
        self.cells[r][c].forbidden = True
        self.cells[r][c].ch = None
        
        if self.symmetry_type == "rotational":
            sym_r, sym_c = self.get_symmetric_cell(r, c)
            if (sym_r, sym_c) != (r, c):
                self.cells[sym_r][sym_c].forbidden = True
                self.cells[sym_r][sym_c].ch = None
        elif self.symmetry_type == "mirror":
            mirror_c = self.cols - 1 - c
            if mirror_c != c:
                self.cells[r][mirror_c].forbidden = True
                self.cells[r][mirror_c].ch = None       

    def count_disconnected_components(self) -> int:
        """Count the number of disconnected white cell components"""
        visited = set()
        components = 0
        
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.is_black(r, c) and (r, c) not in visited:
                    # Start new component
                    components += 1
                    stack = [(r, c)]
                    
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited or self.is_black(cr, cc):
                            continue
                        
                        visited.add((cr, cc))
                        
                        for nr, nc in self.get_neighbors(cr, cc):
                            if (nr, nc) not in visited:
                                stack.append((nr, nc))
    
        return components        

    def count_reachable_white_cells(self, start_r: int, start_c: int) -> int:
        """Count white cells reachable from starting position using flood fill"""
        if self.is_black(start_r, start_c):
            return 0
        
        visited = set()
        stack = [(start_r, start_c)]
        count = 0
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or self.is_black(r, c):
                continue
            
            visited.add((r, c))
            count += 1
            
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) not in visited:
                    stack.append((nr, nc))
        
        return count


    def simulate_word_placement_impact(self, word: str, row: int, col: int, horizontal: bool, available_words: List[str]) -> Dict[str, float]:
        """Simulate impact of word placement on grid connectivity and fillability"""
        scores = {"connectivity": 0, "symmetry": 0, "fillability": 0}
        
        # Create temporary grid copy for simulation
        temp_grid = Grid(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.is_black(r, c):
                    temp_grid.set_letter(r, c, self.letter(r, c) or 'X')
                temp_grid.cells[r][c].forbidden = self.cells[r][c].forbidden
        
        # Place the word temporarily
        dr, dc = (0, 1) if horizontal else (1, 0)
        placed_cells = []
        for i in range(len(word)):
            cr, cc = row + dr * i, col + dc * i
            temp_grid.set_letter(cr, cc, word[i])
            placed_cells.append((cr, cc))
        
        # Check connectivity impact
        components_before = self.count_disconnected_components()
        components_after = temp_grid.count_disconnected_components()
        scores["connectivity"] = max(0, components_before - components_after)  # Reward reducing components
        
        # Check symmetry compatibility
        sym_row, sym_col = temp_grid.get_symmetric_cell(row, col)
        if horizontal:
            sym_horizontal = True  # 180-degree rotation preserves orientation for horizontal
        else:
            sym_horizontal = False
        
        # Check if symmetric position can accommodate a word of same length
        if temp_grid.in_bounds(sym_row, sym_col) and temp_grid.in_bounds(sym_row + dr * (len(word) - 1), sym_col + dc * (len(word) - 1)):
            can_place_symmetric = True
            for i in range(len(word)):
                sym_cr, sym_cc = sym_row + dr * i, sym_col + dc * i
                if temp_grid.is_forbidden(sym_cr, sym_cc):
                    can_place_symmetric = False
                    break
                # Check if there's a conflicting letter (different from what we'd need)
                existing = temp_grid.letter(sym_cr, sym_cc)
                if existing and existing != 'X':  # 'X' is our temporary placeholder
                    can_place_symmetric = False
                    break
            
            scores["symmetry"] = 2.0 if can_place_symmetric else -1.0
        else:
            scores["symmetry"] = -2.0  # Penalty for impossible symmetric placement
        
        return scores


    def check_basic_dead_ends(self, word: str, row: int, col: int, horizontal: bool) -> bool:
        """Enhanced dead-end check with connectivity and symmetry considerations"""
        # Quick basic checks first
        if len(word) < 2:
            return False
        
        # Check bounds
        dr, dc = (0, 1) if horizontal else (1, 0)
        end_r, end_c = row + dr * (len(word) - 1), col + dc * (len(word) - 1)
        if not self.in_bounds(end_r, end_c):
            return False
        
        # Check for forbidden cells
        for i in range(len(word)):
            cr, cc = row + dr * i, col + dc * i
            if self.is_forbidden(cr, cc):
                return False
        
        # Enhanced checks: symmetry compatibility
        if self.symmetry_type == "rotational":
            sym_row, sym_col = self.get_symmetric_cell(row, col)
            sym_end_r, sym_end_c = self.get_symmetric_cell(end_r, end_c)
        elif self.symmetry_type == "mirror":
            sym_row, sym_col = self.get_mirror_cell(row, col)
            sym_end_r, sym_end_c = self.get_mirror_cell(end_r, end_c)
        else:
            # No symmetry checks needed
            return True

        # Ensure symmetric position is also valid
        if not (self.in_bounds(sym_row, sym_col) and self.in_bounds(sym_end_r, sym_end_c)):
            return False

        # Check if symmetric cells are not forbidden
        for i in range(len(word)):
            cr, cc = row + dr * i, col + dc * i
            if self.symmetry_type == "rotational":
                sym_cr, sym_cc = self.get_symmetric_cell(cr, cc)
            elif self.symmetry_type == "mirror":
                sym_cr, sym_cc = self.get_mirror_cell(cr, cc)
            else:
                continue
                
            if (sym_cr, sym_cc) != (cr, cc) and self.is_forbidden(sym_cr, sym_cc):
                return False
        
        return True

    def to_svg(self, cell_size: int = 32, number_font: int = None, show_solution: bool = True, design_box: dict = None) -> str:
        if number_font is None:
            number_font = max(10, int(cell_size * 0.34))
        letter_font = int(cell_size * 0.6)

        width = self.cols * cell_size
        height = self.rows * cell_size
        pad = int(cell_size * 0.1)

        numbering = compute_numbering(self)

        parts = []
        parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        parts.append('<defs><style>')
        parts.append('.cell.black { fill: #000; }')
        parts.append('.cell.white { fill: #fff; stroke: #000; stroke-width: 1px; }')
        parts.append('.design-box { stroke: #000; stroke-width: 2px; }')
        parts.append('.num { font-family: Arial, sans-serif; fill: #000; }')
        parts.append('.letter { font-family: Arial, sans-serif; fill: #333; }')
        parts.append('</style></defs>')

        parts.append('<g id="cells">')
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cell_size
                y = r * cell_size
                if self.is_black(r, c):
                    parts.append(f'<rect id="cell-{r}-{c}" class="cell black" x="{x}" y="{y}" width="{cell_size}" height="{cell_size}"/>')
                else:
                    parts.append(f'<rect id="cell-{r}-{c}" class="cell white" x="{x}" y="{y}" width="{cell_size}" height="{cell_size}"/>')
        parts.append('</g>')

        if design_box and design_box.get('enabled'):
            box_size = design_box.get('size', 3)
            box_color = design_box.get('color', 'black')
            start_row = (self.rows - box_size) // 2
            start_col = (self.cols - box_size) // 2
            parts.append('<g id="design-box">')
            for r in range(start_row, start_row + box_size):
                for c in range(start_col, start_col + box_size):
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        x = c * cell_size
                        y = r * cell_size
                        fill_color = "#000" if box_color == "black" else "#fff"
                        parts.append(f'<rect id="design-cell-{r}-{c}" class="design-box" x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill_color}"/>')
            parts.append('</g>')

        parts.append('<g id="numbers">')
        for (r, c), n in numbering.items():
            x = c * cell_size + pad
            y = r * cell_size + number_font + pad * 0.3
            parts.append(f'<text class="num" x="{x}" y="{y}" font-size="{number_font}">{n}</text>')
        parts.append('</g>')

        if show_solution:
            parts.append('<g id="solution-letters">')
            for r in range(self.rows):
                for c in range(self.cols):
                    ch = self.letter(r, c)
                    if ch:
                        x = c * cell_size + cell_size * 0.5
                        y = r * cell_size + cell_size * 0.7
                        parts.append(f'<text class="letter" x="{x}" y="{y}" font-size="{letter_font}" text-anchor="middle">{escape_xml(ch)}</text>')
            parts.append('</g>')

        # New separate layer for Illustrator workflows
        parts.append('<g id="answers">')
        for r in range(self.rows):
            for c in range(self.cols):
                ch = self.letter(r, c)
                if ch:
                    x = c * cell_size + cell_size * 0.5
                    y = r * cell_size + cell_size * 0.7
                    parts.append(f'<text class="letter" x="{x}" y="{y}" font-size="{letter_font}" text-anchor="middle">{escape_xml(ch)}</text>')
        parts.append('</g>')

        parts.append('</svg>')
        return "".join(parts)

def escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ----------------------------
# Crossword generation logic
# ----------------------------

# ALLOWED_CHARS = set(string.ascii_letters + string.digits + "ÃÃ€Ã‚Ã„ÃƒÃ…Ã†Ã‡Ã‰ÃˆÃŠÃ‹ÃÃŒÃŽÃÃ'Ã"Ã'Ã"Ã–Ã•Ã˜ÃšÃ™Ã›ÃœÃÅ¸ÃžÃŸÄ²Å'Ä³Å"Ã¡Ã Ã¢Ã¤Ã£Ã¥Ã¦Ã§Ã©Ã¨ÃªÃ«Ã­Ã¬Ã®Ã¯Ã±Ã³Ã²Ã´Ã¶ÃµÃ¸ÃºÃ¹Ã»Ã¼Ã½Ã¿Ã¾''-")
ALLOWED_CHARS = set(string.ascii_letters + string.digits + "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿþ'-")

def normalize_answer(s: str) -> str:
    """
    Normaliseer antwoorde. As s nie 'n string is nie, maak dit 'n leë string.
    """
    if s is None:
        s = ""
    else:
        s = str(s)
    s = s.strip().replace(" ", "")
    s = "".join(ch for ch in s if ch in ALLOWED_CHARS)
    return s.upper()

def _get_crossword_letter_bonus() -> Dict[str, float]:
    """
    Klein, beheerde per-letter bonusse vir tipiese Afrikaanse kruiswoord-werkperde.
    Waardes is konserwatief om nie die frekwensie-telling te verdring nie.
    """
    return {
        'E': 0.18,
        'A': 0.16,
        'R': 0.14,
        'S': 0.12,
        # Jy kan later uitbrei met byv. 'N': 0.10, 'I': 0.09, ens.
    }

def calculate_word_flexibility_afrikaans(word: str, all_words: List[Dict[str, str]]) -> float:
    """Verbeterde buigsaamheid met korrekte lang-woord straf, ligte bonusse en kruiswoord-letter bonus."""
    letter_freq = get_afrikaans_letter_frequency()
    letter_bonus = _get_crossword_letter_bonus()
    flexibility = 0.0

    # Basiese frekwensie gewig plus klein per-letter bonus vir E A R S
    for ch in word:
        flexibility += letter_freq.get(ch, 0.001) * 10.0
        flexibility += letter_bonus.get(ch, 0.0)

    vowels = set('AEIOU')
    vowel_positions = [i for i, ch in enumerate(word) if ch in vowels]
    if len(vowel_positions) >= 2:
        flexibility += 2.0
    elif len(vowel_positions) == 1:
        flexibility += 0.6

    # Straf eers baie lank, dan lank
    if len(word) > 12:
        flexibility *= 0.5
    elif len(word) > 10:
        flexibility *= 0.7

    # Bonus vir medium lengtes
    if 5 <= len(word) <= 8:
        flexibility += 1.2

    # Klein bonus vir uiteenlopende letters wat oorvleuelings vergroot
    flexibility += min(1.0, len(set(word)) * 0.03)

    # Ligte ekstra bonus vir woorde wat verskeie van die gewilde letters bevat
    common_set = set(letter_bonus.keys())
    common_count = sum(1 for ch in set(word) if ch in common_set)
    flexibility += min(2.0, 0.35 * common_count)

    return flexibility



def pair_words_by_length(entries: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Pre-pair verskillende woorde met gelyke lengtes vir simmetriese plasing."""
    unique_entries = []
    seen_answers = set()
    for entry in entries:
        if entry['answer'] not in seen_answers:
            unique_entries.append(entry)
            seen_answers.add(entry['answer'])

    length_groups = defaultdict(list)
    for entry in unique_entries:
        length_groups[len(entry['answer'])].append(entry)

    paired, unpaired = [], []
    for length, words in length_groups.items():
        if len(words) < 2:
            unpaired.extend(words)
            continue
        # Sorteer om deterministies te wees en parings te stabiliseer
        words.sort(key=lambda e: e["answer"])
        i = 0
        while i < len(words) - 1:
            if words[i]["answer"] != words[i + 1]["answer"]:
                paired.extend([words[i], words[i + 1]])
                i += 2
            else:
                unpaired.append(words[i])
                i += 1
        if i < len(words):
            unpaired.append(words[i])
    return paired, unpaired




def get_optimal_word_order_afrikaans(entries: List[Dict[str, str]], symmetry_type: str) -> List[Dict[str, str]]:
    """Afrikaans-optimized word ordering with symmetry considerations"""
    # Defensiewe tipe-check
    for e in entries:
        if not isinstance(e.get("answer"), str):
            raise ValueError(f"Word entry 'answer' moet string wees, kry: {e.get('answer')!r}")

    # Remove duplicates while preserving order
    seen_answers = set()
    unique_entries = []
    for entry in entries:
        if entry['answer'] not in seen_answers:
            unique_entries.append(entry)
            seen_answers.add(entry['answer'])

    if symmetry_type == "none":
        unique_entries.sort(key=lambda e: (
            -calculate_word_flexibility_afrikaans(e["answer"], unique_entries),
            -len(e["answer"]),
            random.random()
        ))
        return unique_entries
    
    # Pre-pair DIFFERENT words of equal lengths for symmetric placement
    paired_words, unpaired_words = pair_words_by_length(unique_entries)
    
    # Sort paired words by length, longest first
    paired_words.sort(key=lambda e: len(e["answer"]), reverse=True)
    
    # Sort unpaired words by flexibility/length
    unpaired_words.sort(key=lambda e: (
        -calculate_word_flexibility_afrikaans(e["answer"], unique_entries),
        -len(e["answer"]),
        random.random()
    ))
    
    # Strategic interleaving optimized for symmetry
    ordered = []
    if paired_words:
        ordered.extend(paired_words[:2])
        paired_words = paired_words[2:]

    while paired_words or unpaired_words:
        if len(paired_words) >= 2:
            ordered.extend(paired_words[:2])
            paired_words = paired_words[2:]
        if unpaired_words:
            ordered.append(unpaired_words.pop(0))

    ordered.extend(paired_words)

    # Ligter shuffle om herhalende beginletters te breek
    if ordered:
        from collections import deque, defaultdict
        buckets = defaultdict(deque)
        for e in ordered:
            buckets[e["answer"][0]].append(e)
        keys = list(buckets.keys())
        random.shuffle(keys)
        mixed = []
        while any(buckets.values()):
            for k in keys:
                if buckets[k]:
                    mixed.append(buckets[k].popleft())
        ordered = mixed

    return ordered


def get_afrikaans_letter_frequency():
    """Afrikaans letter frequencies for better word flexibility scoring"""
    # Based on Afrikaans text analysis - common letters first
    return {
        'E': 0.127, 'A': 0.108, 'N': 0.086, 'R': 0.081, 'I': 0.073,
        'T': 0.071, 'S': 0.064, 'O': 0.062, 'L': 0.053, 'D': 0.049,
        'K': 0.045, 'G': 0.041, 'V': 0.038, 'M': 0.037, 'U': 0.035,
        'W': 0.034, 'H': 0.033, 'B': 0.029, 'P': 0.025, 'J': 0.023,
        'F': 0.018, 'C': 0.015, 'Y': 0.010, 'Z': 0.008, 'Q': 0.002, 'X': 0.001
    }

def detect_large_black_regions(grid: Grid) -> List[Tuple[int, int, int]]:
    """Find large connected black regions that could potentially be filled"""
    visited = set()
    large_regions = []
    
    def flood_fill_black(start_r: int, start_c: int) -> int:
        """Count size of connected black region"""
        if (start_r, start_c) in visited:
            return 0
        if not grid.in_bounds(start_r, start_c):
            return 0
        if not grid.is_black(start_r, start_c):
            return 0
            
        visited.add((start_r, start_c))
        size = 1
        
        # Check 4 directions
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = start_r + dr, start_c + dc
            size += flood_fill_black(nr, nc)
        
        return size
    
    # Find all large black regions
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.is_black(r, c) and (r, c) not in visited:
                region_size = flood_fill_black(r, c)
                if region_size >= 9:  # Regions of 9+ black cells are concerning
                    large_regions.append((r, c, region_size))
    
    return large_regions

def try_fill_black_regions(grid: Grid, available_words: List[Dict[str, str]], max_attempts: int = 50) -> List[PlacedWord]:
    """Try to place words in large black regions to improve density"""
    large_regions = detect_large_black_regions(grid)
    if not large_regions:
        return []
    
    additional_placements = []
    used_words = set()
    
    # Focus on smaller words for gap-filling
    gap_fillers = [w for w in available_words if 3 <= len(w['answer']) <= 7]
    gap_fillers.sort(key=lambda w: calculate_word_flexibility_afrikaans(w['answer'], available_words), reverse=True)
    
    for region_r, region_c, region_size in large_regions[:3]:  # Only try top 3 largest
        if len(additional_placements) >= 5:  # Don't over-fill
            break
            
        # Try to place words near or through this region
        attempts = 0
        for word_data in gap_fillers:
            if attempts >= max_attempts:
                break
            if word_data['answer'] in used_words:
                continue
                
            word = word_data['answer']
            
            # Try multiple positions around the black region
            for r_offset in range(-2, 3):
                for c_offset in range(-2, 3):
                    test_r = region_r + r_offset
                    test_c = region_c + c_offset
                    
                    for horizontal in [True, False]:
                        if can_place(grid, word, test_r, test_c, horizontal):
                            # Check if this placement intersects existing words
                            intersections = count_potential_intersections(grid, word, test_r, test_c, horizontal)
                            if intersections > 0:  # Must connect to existing grid
                                place_word(grid, word, test_r, test_c, horizontal)
                                pw = PlacedWord(
                                    answer=word, clue=word_data['clue'],
                                    row=test_r, col=test_c, horizontal=horizontal, length=len(word)
                                )
                                additional_placements.append(pw)
                                used_words.add(word)
                                attempts += 1
                                break
                    if word in used_words:
                        break
                if word in used_words:
                    break
            if word in used_words:
                continue
    
    return additional_placements


def can_place(grid: Grid, word: str, row: int, col: int, horizontal: bool) -> bool:
    # Check for duplicate word placement first
    if is_duplicate_placement(grid, word, row, col, horizontal):
        return False
        
    L = len(word)
    dr, dc = (0, 1) if horizontal else (1, 0)

    # 1. Bounds check for the entire word
    end_r, end_c = row + dr * (L - 1), col + dc * (L - 1)
    if not grid.in_bounds(row, col) or not grid.in_bounds(end_r, end_c):
        return False

    # 2. Check cells before and after the word (to prevent joining words end-to-end)
    before_r, before_c = row - dr, col - dc
    if grid.in_bounds(before_r, before_c) and not grid.is_black(before_r, before_c):
        return False
    after_r, after_c = end_r + dr, end_c + dc
    if grid.in_bounds(after_r, after_c) and not grid.is_black(after_r, after_c):
        return False

    has_intersection = False
    # 3. Iterate through each letter of the word for detailed checks
    for i in range(L):
        cr, cc = row + dr * i, col + dc * i
        
        # 3a. Check for forbidden cells
        if grid.is_forbidden(cr, cc):
            return False

        existing_letter = grid.letter(cr, cc)

        if existing_letter is not None:
            # This is an intersection point
            if existing_letter != word[i]:
                return False  # Conflict: letters don't match
            has_intersection = True
        else:
            # This is an empty cell where a new letter will be placed
            # STRICTURE ADJACENCY RULE: Check perpendicular neighbors to prevent side-by-side words
            perp_neighbors = []
            if horizontal:
                perp_neighbors.extend([(cr - 1, cc), (cr + 1, cc)])
            else: # Vertical
                perp_neighbors.extend([(cr, cc - 1), (cr, cc + 1)])
            
            for nr, nc in perp_neighbors:
                if grid.in_bounds(nr, nc) and not grid.is_black(nr, nc):
                    # Placing this letter would make it lie next to an existing word, which is illegal.
                    return False

    # 4. On a non-empty grid, a new word must connect to at least one existing word.
    if grid.white_count() > 0 and not has_intersection:
        return False

    return True

def place_word(grid: Grid, word: str, row: int, col: int, horizontal: bool):
    """Places a word onto the grid without any automatic symmetry."""
    L = len(word)
    dr, dc = (0, 1) if horizontal else (1, 0)
    for i in range(L):
        cr = row + dr * i
        cc = col + dc * i
        # Always use the simple set_letter; symmetry is handled by the caller
        grid.set_letter(cr, cc, word[i])

def analyze_word_distribution(used_words: List[str]) -> Dict[str, float]:
    """Analyze current distribution of word lengths"""
    if not used_words:
        return {"short": 0, "medium": 0, "long": 0}
    
    short = sum(1 for w in used_words if len(w) <= 4)
    medium = sum(1 for w in used_words if 5 <= len(w) <= 7)
    long = sum(1 for w in used_words if len(w) >= 8)
    total = len(used_words)
    
    return {
        "short": short / total,
        "medium": medium / total, 
        "long": long / total
    }

def grid_component_count(grid) -> int:
    """Tel hoeveel afsonderlike wit-komponente die rooster het."""
    R, C = grid.rows, grid.cols
    seen = [[False]*C for _ in range(R)]

    def is_white(r, c):
        return 0 <= r < R and 0 <= c < C and grid.cells[r][c].ch is not None

    comps = 0
    from collections import deque
    for r in range(R):
        for c in range(C):
            if is_white(r, c) and not seen[r][c]:
                comps += 1
                q = deque([(r, c)])
                seen[r][c] = True
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr + dr, cc + dc
                        if is_white(nr, nc) and not seen[nr][nc]:
                            seen[nr][nc] = True
                            q.append((nr, nc))
    return comps


# Dinamiese teikenverdeling vir woordlengtes per roostergrootte
CURRENT_LEN_TARGETS = {"short": 0.4, "medium": 0.4, "long": 0.2}

def set_length_targets_for_grid(rows: int, cols: int) -> None:
    area = rows * cols
    # Kleiner roosters kry meer kort woorde vir kompakte invul
    if area <= 225:
        CURRENT_LEN_TARGETS.update({"short": 0.45, "medium": 0.40, "long": 0.15})
    elif area <= 400:
        CURRENT_LEN_TARGETS.update({"short": 0.40, "medium": 0.40, "long": 0.20})
    else:
        CURRENT_LEN_TARGETS.update({"short": 0.35, "medium": 0.45, "long": 0.20})


def get_word_length_score(word: str, current_distribution: Dict[str, float]) -> float:
    """Dinamiese lengte-balans telling gebaseer op CURRENT_LEN_TARGETS."""
    length = len(word)
    target = CURRENT_LEN_TARGETS  # gestel deur set_length_targets_for_grid(...)
    category = "short" if length <= 4 else "medium" if length <= 7 else "long"
    shortage = target[category] - current_distribution.get(category, 0.0)
    return max(0.0, shortage * 10.0)


def _make_balanced_batches(clueset: List[Dict[str, str]], batch_count: int) -> List[List[Dict[str, str]]]:
    """
    Verdeel vooraf oor batch_count bondels met balans oor:
    1) lengte 2) beginletters 3) vokale teenwoordigheid
    Voeg dedup by sodat dieselfde leidraad nie twee keer voorkom nie.
    """
    if batch_count <= 0:
        batch_count = 1

    # 0) Dedup bronlys op leidraad en op leidraad+antwoord
    seen_clue = set()
    seen_pair = set()
    deduped: List[Dict[str, str]] = []
    for row in clueset:
        clue_txt = str(row.get("clues", "")).strip().lower()
        ans_raw = str(row.get("answers", "")).strip()
        norm = normalize_answer(ans_raw)
        if len(norm) < 2:
            continue
        pair_key = (clue_txt, norm)
        # Slaan oor as ons reeds die presiese leidraad+antwoord of die leidraad reeds gesien het
        if clue_txt in seen_clue or pair_key in seen_pair:
            continue
        seen_clue.add(clue_txt)
        seen_pair.add(pair_key)
        deduped.append(row)

    # 1) Bou items vir balans
    items = [{"row": row, "norm": normalize_answer(row.get("answers", ""))} for row in deduped]

    # 2) Lengte-buckets
    short  = [it for it in items if len(it["norm"]) <= 4]
    medium = [it for it in items if 5 <= len(it["norm"]) <= 7]
    long   = [it for it in items if len(it["norm"]) >= 8]
    random.shuffle(short); random.shuffle(medium); random.shuffle(long)

    # 3) Maak leë batches
    batches: List[List[Dict[str, str]]] = [[] for _ in range(batch_count)]

    # 4) Round-robin per lengte
    def rr_fill_norm(pool):
        i = 0
        while pool:
            batches[i % batch_count].append(pool.pop())
            i += 1
    rr_fill_norm(long); rr_fill_norm(medium); rr_fill_norm(short)

    # 5) Balans beginletters oor batches
    from collections import Counter, defaultdict, deque
    for _ in range(2):
        letter_counts = [Counter(it["norm"][0] for it in b) for b in batches]
        avg_letter = defaultdict(float)
        for lc in letter_counts:
            for k, v in lc.items():
                avg_letter[k] += v
        for k in list(avg_letter.keys()):
            avg_letter[k] /= max(1, batch_count)
        for bi, b in enumerate(batches):
            over = [it for it in b if letter_counts[bi][it["norm"][0]] > avg_letter[it["norm"][0]] + 1]
            random.shuffle(over)
            for it in over[:2]:
                tgt = min(range(batch_count), key=lambda x: letter_counts[x][it["norm"][0]])
                if tgt != bi:
                    b.remove(it)
                    batches[tgt].append(it)
                    letter_counts[bi][it["norm"][0]] -= 1
                    letter_counts[tgt][it["norm"][0]] += 1

    # 6) Verseker vokale verspreiding
    vowels = set("AEIOU")
    def has_vowel(s: str) -> bool:
        return any(ch in vowels for ch in s)
    for bi, b in enumerate(batches):
        if not b:
            continue
        if sum(1 for it in b if has_vowel(it["norm"])) == 0:
            for bj, other in enumerate(batches):
                if bj == bi:
                    continue
                cand = next((it for it in other if has_vowel(it["norm"])), None)
                if cand:
                    other.remove(cand)
                    b.append(cand)
                    break

    # 7) Strip na oorspronklike rye
    for i in range(len(batches)):
        random.shuffle(batches[i])
        batches[i] = [it["row"] for it in batches[i]]

    return batches



def count_potential_intersections(grid: Grid, word: str, row: int, col: int, horizontal: bool) -> int:
    """Count how many intersections this placement would create"""
    L = len(word)
    dr, dc = (0, 1) if horizontal else (1, 0)
    intersections = 0
    
    for i in range(L):
        cr = row + dr * i
        cc = col + dc * i
        if grid.letter(cr, cc) == word[i]:
            intersections += 1
    
    return intersections

def _vowel_overlap_bonus(grid: "Grid", word: str, row: int, col: int, horizontal: bool) -> float:
    vowels = set("AEIOU")
    dr, dc = (0,1) if horizontal else (1,0)
    bonus = 0.0
    for i, ch in enumerate(word):
        r, c = row + dr*i, col + dc*i
        if grid.in_bounds(r, c) and grid.letter(r, c) == ch and ch in vowels:
            bonus += 0.6  # klein bonus vir vokale-kruisings
    return bonus

def _edge_compactness_bonus(grid: "Grid", word: str, row: int, col: int, horizontal: bool) -> float:
    # bevoordeel plasings nader aan middel om kompakte wit-gebied te hou
    L = len(word)
    dr, dc = (0,1) if horizontal else (1,0)
    center_r, center_c = grid.rows // 2, grid.cols // 2
    word_center_r = row + dr * (L // 2)
    word_center_c = col + dc * (L // 2)
    dist = abs(word_center_r - center_r) + abs(word_center_c - center_c)
    return max(0.0, (grid.rows + grid.cols) - dist) / (grid.rows + grid.cols)


def is_duplicate_placement(grid: Grid, word: str, row: int, col: int, horizontal: bool) -> bool:
    """Check if this word is already placed somewhere else in the grid"""
    for pw in grid.placed:
        if pw.answer == word:
            return True
    return False

def _get_2x2_black_penalty(grid: Grid, word: str, row: int, col: int, horizontal: bool) -> float:
    """Calculates a penalty for creating 2x2 black squares, which can fragment the grid."""
    penalty = 0.0
    L = len(word)
    dr, dc = (0, 1) if horizontal else (1, 0)
    
    # Temporarily identify the cells the new word would occupy
    temp_placed_cells = set()
    for i in range(L):
        cr, cc = row + dr * i, col + dc * i
        temp_placed_cells.add((cr, cc))

    # Check around each cell of the potential word path
    for i in range(L):
        r, c = row + dr * i, col + dc * i
        # Check the four 2x2 squares that this cell (r, c) could be a part of
        for sr in [r - 1, r]:
            for sc in [c - 1, c]:
                if not grid.in_bounds(sr, sc) or not grid.in_bounds(sr + 1, sc + 1):
                    continue
                
                corners = [(sr, sc), (sr + 1, sc), (sr, sc + 1), (sr + 1, sc + 1)]
                is_potential_black_square = True
                for cr, cc in corners:
                    # If a corner is part of the word we are placing, it won't be black
                    if (cr, cc) in temp_placed_cells:
                        is_potential_black_square = False
                        break
                    # If a corner already has a letter, it's not a black square
                    if not grid.is_black(cr, cc):
                        is_potential_black_square = False
                        break
                
                if is_potential_black_square:
                    # This placement would create or complete a 2x2 black square
                    penalty += 3.0
    
    return penalty

def calculate_connectivity_score(grid: Grid, word: str, row: int, col: int, horizontal: bool) -> float:
    """
    Reworked scoring function to more aggressively prioritize density and multiple intersections.
    A higher score is better.
    """
    L = len(word)
    dr, dc = (0, 1) if horizontal else (1, 0)
    
    # --- Helper function for 2x2 white square penalty ---
    # NOTE: This function is now defined INSIDE the main function to resolve the scope error.
    def _two_by_two_white_penalty() -> float:
        penalty = 0.0
        # Create a temporary set of the new word's coordinates for quick lookups
        word_coords = set()
        for i in range(L):
            word_coords.add((row + dr * i, col + dc * i))

        # Check for 2x2 white squares that would be created by placing this word
        for i in range(L):
            r, c = row + dr * i, col + dc * i
            # Check the four 2x2 squares this new cell could be part of
            for r_offset in [-1, 0]:
                for c_offset in [-1, 0]:
                    r0, c0 = r + r_offset, c + c_offset
                    corners = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
                    
                    is_white_square = True
                    for cr, cc in corners:
                        # A cell is "white" if it's part of the new word or already has a letter
                        if (cr, cc) not in word_coords and (not grid.in_bounds(cr, cc) or grid.is_black(cr, cc)):
                            is_white_square = False
                            break
                    
                    if is_white_square:
                        penalty += 2.5
        return penalty

    intersections = count_potential_intersections(grid, word, row, col, horizontal)
    
    # A word must intersect with the grid if the grid is not empty.
    if grid.white_count() > 0 and intersections == 0:
        return -1e9  # Effectively impossible

    # --- Core Score: Based heavily on number of intersections ---
    # Exponentially reward multiple intersections to create dense clusters.
    # 1 intersection is the baseline. 2 is much better. 3+ is fantastic.
    if intersections >= 3:
        score = 100.0 + (intersections - 3) * 25.0
    elif intersections == 2:
        score = 60.0
    elif intersections == 1:
        score = 25.0
    else: # Should only happen on an empty grid
        score = 5.0

    # --- Bonuses ---
    # Bonus for being close to the center to encourage a central mass
    score += _edge_compactness_bonus(grid, word, row, col, horizontal) * 5.0
    
    # Bonus for crossing over vowels, as they are good connecting letters
    score += _vowel_overlap_bonus(grid, word, row, col, horizontal) * 3.0
    
    # Bonus for creating new potential crossing points.
    potential_crossings = 0
    for i in range(L):
        cr, cc = row + dr * i, col + dc * i
        # Only count potential for new letters, not on existing intersections
        if grid.letter(cr, cc) is None:
            # Check perpendicular neighbors
            perp_dr, perp_dc = (1, 0) if horizontal else (0, 1)
            for sign in [-1, 1]:
                nr, nc = cr + perp_dr * sign, cc + perp_dc * sign
                if grid.in_bounds(nr, nc) and grid.is_black(nr, nc):
                    potential_crossings += 1
    score += potential_crossings * 1.5

    # --- Penalties ---
    # Heavy penalty for creating 2x2 white squares ("holes")
    score -= _two_by_two_white_penalty() * 15.0
    
    # Penalty for creating 2x2 black squares, which fragments the board
    score -= _get_2x2_black_penalty(grid, word, row, col, horizontal)

    return score



def find_best_position_optimized(grid: Grid, word: str, current_distribution: Dict[str, float], max_candidates: int = 20, design_box: dict = None) -> Optional[Tuple[int, int, bool, float]]:
    """Optimized version with candidate limiting to improve performance + shared-letter fast path"""
    candidates = []
    L = len(word)
    if L == 0:
        return None

    # As die rooster leeg is, probeer sinvolle beginplekke
    if grid.white_count() == 0:
        midr, midc = grid.rows // 2, grid.cols // 2
        if design_box and design_box.get('enabled'):
            box_size = design_box.get('size', 3)
            box_start_r = (grid.rows - box_size) // 2
            box_start_c = (grid.cols - box_size) // 2
            test_positions = [
                (box_start_r - 2, midc, True),
                (box_start_r - 2, midc, False),
                (midr, box_start_c - 2, True),
                (midr, box_start_c - 2, False),
                (box_start_r + box_size + 2, midr, True),
                (box_start_r + box_size + 2, midr, False),
                (midr, box_start_c + box_size + 2, True),
                (midr, box_start_c + box_size + 2, False),
            ]
            for test_r, test_c, horiz in test_positions:
                if horiz:
                    start_c = max(0, min(grid.cols - L, test_c - L // 2))
                    if can_place(grid, word, test_r, start_c, True):
                        return (test_r, start_c, True, 10.0)
                else:
                    start_r = max(0, min(grid.rows - L, test_r - L // 2))
                    if can_place(grid, word, start_r, test_c, False):
                        return (start_r, test_c, False, 10.0)
        else:
            start_c = max(0, min(grid.cols - L, midc - L // 2))
            if can_place(grid, word, midr, start_c, True):
                return (midr, start_c, True, 10.0)
            start_r = max(0, min(grid.rows - L, midr - L // 2))
            if can_place(grid, word, start_r, midc, False):
                return (start_r, midc, False, 10.0)
        return None

    # Rooster is nie leeg nie: stel vinnig vas of die woord en rooster enige letter deel
    existing_coords = [(r, c, grid.letter(r, c))
                       for r in range(grid.rows)
                       for c in range(grid.cols)
                       if grid.letter(r, c) is not None]
    grid_letters = {ch for _, _, ch in existing_coords}
    if not any(ch in grid_letters for ch in word):
        # Geen gedeelde letters nie, laat dit vir 'n latere siklus
        return None


    # For non-empty grids - OPTIMIZED approach
    existing_coords = [(r, c, grid.letter(r, c))
                       for r in range(grid.rows)
                       for c in range(grid.cols)
                       if grid.letter(r, c) is not None]

    # Build letter position map
    letter_positions: Dict[str, List[Tuple[int, int]]] = {}
    for r, c, ch in existing_coords:
        letter_positions.setdefault(ch, []).append((r, c))

    # Method 1: Try intersections (with early stopping)
    # Adaptiewe kandidaatlimiet
    area = grid.rows * grid.cols
    base = 14 if area <= 225 else 20 if area <= 400 else 28
    local_max_candidates = max(12, min(base + max(0, L - 6), 36))

    # Rangskik letters volgens hoe veel hulle reeds op die rooster is
    letter_counts = {ch: len(letter_positions.get(ch, [])) for ch in set(word)}
    ordered_letters = sorted(set(word), key=lambda ch: letter_counts.get(ch, 0), reverse=True)

    # Sorteer die posisies vir elke letter nader aan middelpunt
    midr, midc = grid.rows // 2, grid.cols // 2

    candidates_found = 0
    for ch in ordered_letters:
        if ch not in letter_positions or candidates_found >= local_max_candidates:
            continue

        pos_sorted = sorted(letter_positions[ch], key=lambda rc: abs(rc[0]-midr) + abs(rc[1]-midc))

        # Gebruik die eerste voorkoms-indeks van hierdie letter in die woord vir uitlijning
        i = word.index(ch)

        for (r, c) in pos_sorted:
            if candidates_found >= local_max_candidates:
                break

            for horiz in [True, False]:
                if candidates_found >= local_max_candidates:
                    break

                start_r = r if horiz else r - i
                start_c = c - i if horiz else c

                if can_place(grid, word, start_r, start_c, horiz):
                    connectivity_score = calculate_connectivity_score(grid, word, start_r, start_c, horiz)
                    if connectivity_score > 1.0:
                        length_bonus = get_word_length_score(word, current_distribution)
                        # ekstra sagte voorkeur vir ≥2 kruisings om kompakter te word
                        shared = count_potential_intersections(grid, word, start_r, start_c, horiz)
                        extra = 3.0 if shared >= 2 else 0.0
                        total_score = connectivity_score + length_bonus + extra + max(0.0, (8 - L) * 0.05) + random.random() * 0.5
                        candidates.append((start_r, start_c, horiz, total_score))
                        candidates_found += 1

    # If we have good candidates, use them
    if len(candidates) >= 3:
        candidates.sort(key=lambda t: t[3], reverse=True)
        top_count = min(5, len(candidates))
        return random.choice(candidates[:top_count])

    # Method 2: Quick fallback for hard-to-place words
    # Try a smaller sample of positions near existing words
    sample_coords = random.sample(existing_coords, min(10, len(existing_coords)))
    
    for r, c, _ in sample_coords:
        for offset in [-2, -1, 1, 2]:  # Reduced range
            for horiz in [True, False]:
                if horiz:
                    test_r, test_c = r, c + offset
                else:
                    test_r, test_c = r + offset, c
                
                if can_place(grid, word, test_r, test_c, horiz):
                    connectivity_score = calculate_connectivity_score(grid, word, test_r, test_c, horiz)
                    if connectivity_score > 0:
                        total_score = connectivity_score + random.random() * 0.5
                        candidates.append((test_r, test_c, horiz, total_score))
                        
                        # Early exit for performance
                        if len(candidates) >= 3:
                            break
            if len(candidates) >= 3:
                break
        if len(candidates) >= 3:
            break

    if not candidates:
        return None
    
    # Pick best from available candidates
    candidates.sort(key=lambda t: t[3], reverse=True)
    return candidates[0]

def update_progress(task_id: str, progress: int, status: str):
    """Update progress for a task"""
    PROGRESS_STORE[task_id] = {
        "progress": progress,
        "status": status,
        "timestamp": time.time()
    }

@dataclass
class PlacementState:
    """Represents a state in the backtracking search"""
    grid_state: List[List[Cell]]  # Deep copy of grid cells
    placed_words: List[PlacedWord]  # List of placed words at this state
    remaining_words: List[Dict[str, str]]  # Words still to be placed
    placement_history: List[Tuple[str, int, int, bool]]  # (word, row, col, horizontal)

def deep_copy_grid_state(grid: Grid) -> List[List[Cell]]:
    """Create a deep copy of the grid's cell state"""
    return [[Cell(ch=cell.ch, forbidden=cell.forbidden) for cell in row] for row in grid.cells]

def restore_grid_state(grid: Grid, saved_state: List[List[Cell]]):
    """Restore grid to a previous state"""
    for r in range(grid.rows):
        for c in range(grid.cols):
            grid.cells[r][c] = Cell(ch=saved_state[r][c].ch, forbidden=saved_state[r][c].forbidden)

def can_make_progress(grid: Grid, remaining_words: List[Dict[str, str]], max_test_words: int = 10) -> bool:
    """Quick check if any of the remaining words can potentially be placed"""
    test_words = remaining_words[:max_test_words]  # Only test first few words for speed
    
    for word_data in test_words:
        word = word_data['answer']
        # Try a few strategic positions
        for r in range(0, grid.rows, 2):  # Sample every other row
            for c in range(0, grid.cols, 2):  # Sample every other column
                for horizontal in [True, False]:
                    if can_place(grid, word, r, c, horizontal):
                        return True
    return False

def generate_crossword_with_backtracking(
    clueset: List[Dict[str, str]],
    rows: int,
    cols: int,
    target_black_ratio: float,
    max_attempts: int = 50,  # Not used in single-run logic
    rng_seed: Optional[int] = None,
    task_id: Optional[str] = None,
    design_box: Optional[dict] = None,
    symmetry_type: str = "none",
    strict_symmetry: bool = False,
    max_backtracks: int = 50
) -> Tuple[Grid, List[PlacedWord], List[Dict[str, str]]]:
    """Crossword generation with backtracking capability (Corrected Version)."""
    # Hard type coercion en sanity checks om str vs int vergelykings te voorkom
    try:
        rows = int(rows)
        cols = int(cols)
    except Exception:
        raise ValueError(f"rows/cols moet ints wees. Ontvang rows={rows!r}, cols={cols!r}")
    try:
        target_black_ratio = float(target_black_ratio)
    except Exception:
        target_black_ratio = 0.35  # veilige verstek

    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows en cols moet > 0 wees. Ontvang rows={rows}, cols={cols}")
    if not (0.0 <= target_black_ratio <= 0.9):
        target_black_ratio = max(0.0, min(0.9, target_black_ratio))

    set_length_targets_for_grid(rows, cols)

    if rng_seed is not None:
        random.seed(rng_seed)
    else:
        random.seed()

    if task_id:
        update_progress(task_id, 5, "Analyzing CSV entries with backtracking...")
    
    entries = []
    seen_clue = set()
    seen_pair = set()
    for i, row in enumerate(clueset):
        clue = (row.get("clues") or "").strip()
        ans_raw = (row.get("answers") or "").strip()
        norm = normalize_answer(ans_raw)
        if len(norm) >= 2:
            key_clue = clue.lower()
            key_pair = (key_clue, norm)
            if key_clue not in seen_clue and key_pair not in seen_pair:
                entries.append({"clue": clue, "answer": norm, "raw_answer": ans_raw})
                seen_clue.add(key_clue)
                seen_pair.add(key_pair)

    if task_id:
        update_progress(task_id, 10, "Starting backtracking generation...")

    # Skoonmaak vir backtracking pad ook
    clean_entries = []
    for e in entries:
        if not isinstance(e.get("clue"), str):
            app.logger.warning(f"Skip clue nie-string: {e.get('clue')!r}")
            continue
        if not isinstance(e.get("answer"), str):
            app.logger.warning(f"Skip answer nie-string: {e.get('answer')!r}")
            continue
        clean_entries.append(e)
    entries = clean_entries

    grid = Grid(rows, cols, symmetry_type, strict_symmetry)
    sorted_words = get_optimal_word_order_afrikaans(entries, symmetry_type)
    
    if design_box and design_box.get("enabled"):
        box_size = design_box.get("size", 3)
        start_row = (rows - box_size) // 2
        start_col = (cols - box_size) // 2
        for r in range(start_row, start_row + box_size):
            for c in range(start_col, start_col + box_size):
                if 0 <= r < rows and 0 <= c < cols:
                    grid.set_forbidden_with_symmetry(r, c)

    state_stack: List[PlacementState] = []
    used: List[PlacedWord] = []
    unused_words = sorted_words.copy()
    backtrack_count = 0

    if unused_words:
        first_word_data = unused_words.pop(0)
        word = first_word_data['answer']
        placement = find_best_position_optimized(grid, word, {"short": 0, "medium": 0, "long": 0}, design_box=design_box)
        if placement:
            r, c, horiz, _ = placement
            place_word(grid, word, r, c, horiz)
            pw = PlacedWord(answer=word, clue=first_word_data['clue'], row=r, col=c, horizontal=horiz, length=len(word))
            grid.placed.append(pw)
            used.append(pw)
            
            state_stack.append(PlacementState(
                grid_state=deep_copy_grid_state(grid),
                placed_words=used.copy(),
                remaining_words=unused_words.copy(),
                placement_history=[(word, r, c, horiz)]
            ))

    while unused_words and backtrack_count < max_backtracks:
        made_progress = False
        words_to_try_this_turn = unused_words.copy()
        
        for word_data in words_to_try_this_turn:
            word_to_place = word_data['answer']
            current_dist = analyze_word_distribution([pw.answer for pw in used])
            placement = find_best_position_optimized(grid, word_to_place, current_dist, design_box=design_box)
            
            if placement:
                r, c, horiz, _ = placement
                
                # Save state before making a move
                state_stack.append(PlacementState(
                    grid_state=deep_copy_grid_state(grid),
                    placed_words=used.copy(),
                    remaining_words=unused_words.copy(),
                    placement_history=[(pw.answer, pw.row, pw.col, pw.horizontal) for pw in used]
                ))
                
                place_word(grid, word_to_place, r, c, horiz)
                pw = PlacedWord(answer=word_to_place, clue=word_data['clue'], row=r, col=c, horizontal=horiz, length=len(word_to_place))
                grid.placed.append(pw)
                used.append(pw)
                unused_words.remove(word_data)
                
                made_progress = True
                break

        if not made_progress and state_stack:
            if task_id:
                update_progress(task_id, 50, f"Backtracking... ({backtrack_count + 1}/{max_backtracks})")
            
            prev_state = state_stack.pop()
            restore_grid_state(grid, prev_state.grid_state)
            grid.placed = prev_state.placed_words.copy()
            used = prev_state.placed_words.copy()
            unused_words = prev_state.remaining_words.copy()
            backtrack_count += 1
            
            if unused_words:
                # Try the next word in the list next time to avoid getting stuck
                problematic_word = unused_words.pop(0)
                unused_words.append(problematic_word)

        elif not made_progress:
            break

    if unused_words:
        added_post = _post_fill_gap_optimization(grid, [{"answers": e["answer"], "clues": e["clue"]} for e in unused_words])
        if added_post:
            used.extend(added_post)
            grid.placed.extend(added_post)
            
    _post_black_mask(grid, target_black_ratio, symmetry_type)

    used_set = set((pw.clue, pw.answer) for pw in used)
    leftovers = [e for e in entries if (e["clue"], e["answer"]) not in used_set]

    if task_id:
        update_progress(task_id, 90, "Backtracking generation complete!")

    return grid, used, leftovers

def generate_crossword(
    clueset: List[Dict[str, str]],
    rows: int,
    cols: int,
    target_black_ratio: float,
    max_attempts: int = 200,
    rng_seed: Optional[int] = None,
    task_id: Optional[str] = None,
    design_box: Optional[dict] = None,
    symmetry_type: str = "none",
    strict_symmetry: bool = False
) -> Tuple[Grid, List[PlacedWord], List[Dict[str, str]]]:
    # Hard type coercion en sanity checks
    try:
        rows = int(rows)
        cols = int(cols)
    except Exception:
        raise ValueError(f"rows/cols moet ints wees. Ontvang rows={rows!r}, cols={cols!r}")
    try:
        target_black_ratio = float(target_black_ratio)
    except Exception:
        target_black_ratio = 0.35

    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows en cols moet > 0 wees. Ontvang rows={rows}, cols={cols}")
    if not (0.0 <= target_black_ratio <= 0.9):
        target_black_ratio = max(0.0, min(0.9, target_black_ratio))

    set_length_targets_for_grid(rows, cols)
    if rng_seed is not None:
        random.seed(rng_seed)
    else:
        random.seed()

    if task_id:
        update_progress(task_id, 5, "Analyzing CSV entries...")
    
    # Shuffle entries; pre-normalize answers, keep mapping back to original clues
    entries = []
    for i, row in enumerate(clueset):
        clue = (row.get("clues") or "").strip()
        ans_raw = (row.get("answers") or "").strip()
        norm = normalize_answer(ans_raw)
        if len(norm) >= 2:  # Avoid 1-letter words
            entries.append({"clue": clue, "answer": norm, "raw_answer": ans_raw})
        
        # Update progress during CSV processing
        if task_id and i % 50 == 0:
            progress = 5 + (i / len(clueset)) * 10
            update_progress(task_id, int(progress), f"Processed {i}/{len(clueset)} entries...")

    if not entries:
        raise ValueError("No valid entries with columns 'clues' and 'answers' (min 2 letters after normalization).")

    # Verwyder enige inskrywings met nie-string leidrade of antwoorde, en log dit
    clean_entries = []
    for e in entries:
        if not isinstance(e.get("clue"), str):
            app.logger.warning(f"Skip clue nie-string: {e.get('clue')!r}")
            continue
        if not isinstance(e.get("answer"), str):
            app.logger.warning(f"Skip answer nie-string: {e.get('answer')!r}")
            continue
        clean_entries.append(e)
    entries = clean_entries

    if task_id:
        update_progress(task_id, 15, f"Found {len(entries)} valid entries. Starting grid generation...")

    # Multiple tries to hit black ratio by varying order and starting word
    best: Optional[Tuple[Grid, List[PlacedWord], List[Dict[str, str]]]] = None
    best_diff = 1.0

    for attempt in range(max_attempts):
        if task_id:
            progress = 15 + (attempt / max_attempts) * 70
            update_progress(task_id, int(progress), f"Attempt {attempt + 1}/{max_attempts}: Shuffling and placing words...")
        
        # Use the updated smart word ordering, passing the symmetry type
        sorted_words = get_optimal_word_order_afrikaans(entries, symmetry_type)

        grid = Grid(rows, cols, symmetry_type, strict_symmetry)
        
        # Pre-mark design box area as forbidden BEFORE any word placement
        if design_box and design_box.get("enabled"):
            box_size = design_box.get("size", 3)
            start_row = (rows - box_size) // 2
            start_col = (cols - box_size) // 2
            
            # Ensure the design box is compatible with symmetry types
            if symmetry_type == "mirror":
                start_col = (cols - box_size) // 2
            elif symmetry_type == "rotational":
                pass # Rotational is naturally compatible with a centered box
            
            for r in range(start_row, start_row + box_size):
                for c in range(start_col, start_col + box_size):
                    if 0 <= r < rows and 0 <= c < cols:
                        # Since we simplified place_word, we must manually handle symmetry here
                        grid.set_forbidden(r,c)
                        if symmetry_type == 'rotational':
                            sym_r, sym_c = grid.get_symmetric_cell(r, c)
                            grid.set_forbidden(sym_r, sym_c)
                        elif symmetry_type == 'mirror':
                            sym_r, sym_c = grid.get_mirror_cell(r, c)
                            grid.set_forbidden(sym_r, sym_c)

        
        used: List[PlacedWord] = []
        unused_words = sorted_words.copy()

        # Handle the very first word placement, which is always unpaired
        if unused_words:
            first_word_data = unused_words.pop(0)
            word = first_word_data['answer']
            placement = find_best_position_optimized(grid, word, {"short": 0, "medium": 0, "long": 0}, design_box=design_box)
            if placement:
                r, c, horiz, _ = placement
                place_word(grid, word, r, c, horiz)
                pw = PlacedWord(answer=word, clue=first_word_data['clue'], row=r, col=c, horizontal=horiz, length=len(word))
                grid.placed.append(pw)
                used.append(pw)

       # Main placement loop - now handles pairs with strict symmetry option
        while unused_words:
            word_to_place_1 = unused_words.pop(0)
            word_to_place_2 = None
            
            # Check if this word can be part of a symmetric pair
            if symmetry_type != "none" and unused_words:
                if len(word_to_place_1['answer']) == len(unused_words[0]['answer']):
                    word_to_place_2 = unused_words.pop(0)

            # --- Placement logic ---
            word_1 = word_to_place_1['answer']
            current_dist = analyze_word_distribution([pw.answer for pw in used])
            placement_1 = find_best_position_optimized(grid, word_1, current_dist, design_box=design_box)
            
            if not placement_1:
                continue # Can't place this word, move to next

            r1, c1, horiz1, _ = placement_1

            # Handle symmetry requirements
            if word_to_place_2 and symmetry_type != "none":
                word_2 = word_to_place_2['answer']
                
                # Calculate symmetric position
                if symmetry_type == "rotational":
                    end_r1, end_c1 = r1 + (len(word_1) - 1) * (0 if horiz1 else 1), c1 + (len(word_1) - 1) * (1 if horiz1 else 0)
                    sym_r, sym_c = grid.get_symmetric_cell(end_r1, end_c1)
                    sym_horiz = horiz1
                else: # mirror
                    sym_r, sym_c = grid.get_mirror_cell(r1, c1)
                    sym_horiz = horiz1

                # Check if symmetric placement is valid
                can_place_symmetric = can_place(grid, word_2, sym_r, sym_c, sym_horiz)
                
                if can_place_symmetric:
                    # Success! Place both words.
                    place_word(grid, word_1, r1, c1, horiz1)
                    pw1 = PlacedWord(answer=word_1, clue=word_to_place_1['clue'], row=r1, col=c1, horizontal=horiz1, length=len(word_1))
                    grid.placed.append(pw1)
                    used.append(pw1)

                    place_word(grid, word_2, sym_r, sym_c, sym_horiz)
                    pw2 = PlacedWord(answer=word_2, clue=word_to_place_2['clue'], row=sym_r, col=sym_c, horizontal=sym_horiz, length=len(word_2))
                    grid.placed.append(pw2)
                    used.append(pw2)
                elif strict_symmetry:
                    # STRICT SYMMETRY: If we can't place both, don't place either
                    unused_words.insert(0, word_to_place_2)  # Put word_2 back
                    # Don't place word_1 either - skip this pair entirely
                    continue
                else:
                    # NON-STRICT: Place word_1 and requeue word_2
                    place_word(grid, word_1, r1, c1, horiz1)
                    pw1 = PlacedWord(answer=word_1, clue=word_to_place_1['clue'], row=r1, col=c1, horizontal=horiz1, length=len(word_1))
                    grid.placed.append(pw1)
                    used.append(pw1)
                    unused_words.insert(0, word_to_place_2)
            elif symmetry_type != "none" and strict_symmetry:
                # STRICT SYMMETRY: Single word in symmetric mode - only allow if it's self-symmetric
                if symmetry_type == "rotational":
                    mid_r = r1 + (len(word_1) - 1) * (0 if horiz1 else 1) / 2
                    mid_c = c1 + (len(word_1) - 1) * (1 if horiz1 else 0) / 2
                    sym_mid_r, sym_mid_c = grid.get_symmetric_cell(int(mid_r), int(mid_c))
                    is_self_symmetric = abs(mid_r - sym_mid_r) < 0.5 and abs(mid_c - sym_mid_c) < 0.5
                else: # mirror
                    mid_c = c1 + (len(word_1) - 1) * (1 if horiz1 else 0) / 2
                    is_self_symmetric = abs(mid_c - (grid.cols - 1 - mid_c)) < 0.5
                
                if is_self_symmetric:
                    place_word(grid, word_1, r1, c1, horiz1)
                    pw1 = PlacedWord(answer=word_1, clue=word_to_place_1['clue'], row=r1, col=c1, horizontal=horiz1, length=len(word_1))
                    grid.placed.append(pw1)
                    used.append(pw1)
                # else: skip this word in strict mode
            else:
                # No symmetry constraints or non-strict mode - place normally
                place_word(grid, word_1, r1, c1, horiz1)
                pw1 = PlacedWord(answer=word_1, clue=word_to_place_1['clue'], row=r1, col=c1, horizontal=horiz1, length=len(word_1))
                grid.placed.append(pw1)
                used.append(pw1)

            if task_id and len(used) % 5 == 0:
                progress_percent = len(used) / len(entries) if entries else 0
                update_progress(task_id, int(progress + progress_percent * 10), f"Attempt {attempt + 1}: Placed {len(used)} words...")


        if task_id:
            update_progress(task_id, 85, "Filling remaining gaps...")  

        if unused_words:
            _ = _post_fill_gap_optimization(grid, [{"answers": w["answer"], "clues": w["clue"]} for w in unused_words])


        # Compute achieved black ratio
        if task_id:
            update_progress(task_id, int(progress) + 12, f"Attempt {attempt + 1}: Evaluating grid quality...")
            
        total = rows * cols
        whites = grid.white_count()
        blacks = total - whites
        achieved = blacks / total
        diff = abs(achieved - target_black_ratio)

        # Keep best, with bonus for having good word distribution
        placed_answers = [pw.answer for pw in used]
        final_dist = analyze_word_distribution(placed_answers)
        balance_score = 1.0 - abs(final_dist["short"] - 0.4) - abs(final_dist["medium"] - 0.4) - abs(final_dist["long"] - 0.2)
        
        # Combined score: ratio accuracy + word balance
        combined_score = (1.0 - diff) * 0.7 + balance_score * 0.3
        
        if best is None or combined_score > (1.0 - best_diff) * 0.7:
            used_set = set((pw.clue, pw.answer) for pw in used)
            leftovers = [e for e in entries if (e["clue"], e["answer"]) not in used_set]
            best = (grid, used, leftovers)
            best_diff = diff
            
            if task_id:
                update_progress(task_id, int(progress) + 13, 
                               f"New best! Placed {len(used)} words, black ratio: {achieved:.1%}")
            
            # Early exit if we have a good solution
            if diff <= 0.05 and balance_score > 0.8:
                if task_id:
                    update_progress(task_id, 85, "Found excellent solution! Finalizing...")
                break

    if task_id:
        update_progress(task_id, 90, "Generation complete! Building reports...")

    assert best is not None
    return best

def categorize_used(placed: List[PlacedWord]) -> Dict[str, List[Dict[str, Any]]]:
    horiz = []
    vert = []
    for pw in placed:
        item = {
            "clue": pw.clue,
            "answer": pw.answer,
            "row": pw.row,
            "col": pw.col,
            "length": pw.length,
            "number": None,  # will be computed by numbering pass below
        }
        if pw.horizontal:
            horiz.append(item)
        else:
            vert.append(item)
    return {"Horizontal": horiz, "Vertical": vert}

def compute_numbering(grid: Grid) -> Dict[Tuple[int, int], int]:
    numbering = {}
    num = 1
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.is_black(r, c):
                continue
            start_across = (c == 0 or grid.is_black(r, c - 1)) and (c + 1 < grid.cols and not grid.is_black(r, c + 1))
            start_down = (r == 0 or grid.is_black(r - 1, c)) and (r + 1 < grid.rows and not grid.is_black(r + 1, c))
            if start_across or start_down:
                numbering[(r, c)] = num
                num += 1
    return numbering

def _score_black_candidate(grid: Grid, r: int, c: int) -> float:
    """Hoër is beter. Kies rande en plekke met baie swart bure om gate toe te maak."""
    if not grid.in_bounds(r, c) or not grid.is_black(r, c):
        return -1e9
    rows, cols = grid.rows, grid.cols
    center_r, center_c = rows // 2, cols // 2
    # afstand van middel help om rande te vul
    dist = abs(r - center_r) + abs(c - center_c)
    # tel swart bure om gate te sluit
    black_nb = 0
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r + dr, c + dc
        if grid.in_bounds(nr, nc) and grid.is_black(nr, nc):
            black_nb += 1
    # bonus as op rand
    on_edge = int(r in (0, rows-1) or c in (0, cols-1))
    return dist * 0.6 + black_nb * 1.0 + on_edge * 0.5


def _post_black_mask(grid: Grid, target_black_ratio: float, symmetry_type: str = "none"):
    """
    Vul ekstra swartblokke om nader aan target_black_ratio te kom,
    en hou 180° of spieël simmetrie as dit versoek is.
    Raak slegs leë selle, nooit letters nie.
    """
    total = grid.rows * grid.cols

    def black_ratio() -> float:
        whites = grid.white_count()
        blacks = total - whites
        return blacks / total

    # Vinnige simmetrie-sanitasie: maak pare leë as een kant reeds swart is
    if symmetry_type in {"rotational", "mirror"}:
        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid.is_black(r, c):
                    if symmetry_type == "rotational":
                        sr, sc = grid.get_symmetric_cell(r, c)
                    else:
                        sr, sc = grid.get_mirror_cell(r, c)
                    # los letters uit
                    if grid.in_bounds(sr, sc) and grid.is_black(sr, sc):
                        # albei reeds swart, niks om te doen nie
                        pass
                    elif grid.in_bounds(sr, sc) and grid.letter(sr, sc) is None:
                        grid.set_forbidden(sr, sc)

    # Voeg swartblokke by totdat ons by die teiken uitkom
    # Kies altyd die beste kandidaat en sy simmetriese maat
    guard = 0
    while black_ratio() + 1e-9 < target_black_ratio and guard < total:
        guard += 1
        best = None
        best_score = -1e9
        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid.letter(r, c) is None:  # leë sel
                    if symmetry_type == "rotational":
                        sr, sc = grid.get_symmetric_cell(r, c)
                    elif symmetry_type == "mirror":
                        sr, sc = grid.get_mirror_cell(r, c)
                    else:
                        sr, sc = r, c

                    # moenie letters aanraak nie
                    if not grid.in_bounds(sr, sc):
                        continue
                    if grid.letter(sr, sc) is not None:
                        continue

                    score = _score_black_candidate(grid, r, c)
                    if symmetry_type in {"rotational", "mirror"}:
                        score += _score_black_candidate(grid, sr, sc)

                    if score > best_score:
                        best_score = score
                        best = (r, c, sr, sc)

        if not best:
            break  # niks sinvol om te kleur nie

        r, c, sr, sc = best
        grid.set_forbidden(r, c)
        grid.set_forbidden(sr, sc)


def _post_fill_gap_optimization(grid: Grid, remaining_entries: List[Dict[str, str]], max_added: int = 6) -> List[PlacedWord]:
    candidates = [{"answer": normalize_answer(e.get("answers") or e.get("answer","")), "clue": e.get("clues") or e.get("clue","")}
                  for e in remaining_entries]
    candidates = [e for e in candidates if 3 <= len(e["answer"]) <= 7]
    added = try_fill_black_regions(grid, candidates, max_attempts=30)
    return added[:max_added]




def build_clue_report(grid: Grid, placed: List[PlacedWord], leftovers: List[Dict[str, str]]) -> Tuple[str, str]:
    numbering = compute_numbering(grid)

    seen_keys = set()
    horiz_list = []
    vert_list = []

    for pw in placed:
        key = (pw.row, pw.col, pw.horizontal)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        num = numbering.get((pw.row, pw.col))
        if not num:
            continue

        entry = {
            "number": num,
            "row": pw.row,
            "col": pw.col,
            "length": pw.length,
            "clue": pw.clue,
            "answer": pw.answer,
        }
        if pw.horizontal:
            horiz_list.append(entry)
        else:
            vert_list.append(entry)

    # Basiese sorteer volgens roosternommer
    horiz_list.sort(key=lambda e: e["number"])
    vert_list.sort(key=lambda e: e["number"])

    # Ligte randomisasie sonder om die orde heeltemal te breek
    # Ons voeg 'n klein "jitter" by die nommer en sorteer weer
    def _slightly_shuffle(items: List[Dict[str, Any]], jitter: float = 0.22) -> List[Dict[str, Any]]:
        # minder as of gelyk aan 3 inskrywings bly in vaste volgorde
        if len(items) <= 3:
            return items
        keyed = []
        for idx, e in enumerate(items):
            # Behou stabiliteit met 'idx' as ties, en verskuif effens met jitter
            k = e["number"] + (random.random() - 0.5) * jitter
            keyed.append((k, idx, e))
        keyed.sort()
        return [e for _, __, e in keyed]

    horiz_list = _slightly_shuffle(horiz_list)
    vert_list = _slightly_shuffle(vert_list)

    # Meng die leftovers effens ook, maar nie wild nie
    if leftovers:
        random.shuffle(leftovers)

    # Tekstrapport
    lines = []
    lines.append("Horizontal")
    for e in horiz_list:
        lines.append(f'{e["number"]}. {e["clue"]} ({len(e["answer"])})')
    lines.append("")
    lines.append("Vertical")
    for e in vert_list:
        lines.append(f'{e["number"]}. {e["clue"]} ({len(e["answer"])})')
    lines.append("")
    lines.append("Leftovers")
    for e in leftovers:
        lines.append(f'- {e["clue"]} [{e["answer"]}]')
    text_report = "\n".join(lines)

    # CSV
    csv_out = io.StringIO()
    writer = csv.writer(csv_out)
    writer.writerow(["category", "number", "row", "col", "length", "clues", "answers"])

    # Lookup vir posisie-inligting
    number_lookup = {}
    for pw in placed:
        num = numbering.get((pw.row, pw.col))
        if num:
            number_lookup[(num, pw.horizontal)] = (pw.row, pw.col, pw.length)

    for e in horiz_list:
        row, col, length = number_lookup.get((e["number"], True), ("", "", len(e["answer"])))
        writer.writerow(["Horizontal", e["number"], row, col, length, e["clue"], e["answer"]])
    for e in vert_list:
        row, col, length = number_lookup.get((e["number"], False), ("", "", len(e["answer"])))
        writer.writerow(["Vertical", e["number"], row, col, length, e["clue"], e["answer"]])
    for e in leftovers:
        writer.writerow(["Leftovers", "", e["clue"], e["answer"]])

    return text_report, csv_out.getvalue()



def _post_fill_gap_optimization(grid: Grid, remaining_entries: List[Dict[str, str]], max_added: int = 6) -> List[PlacedWord]:
    """
    Vul groot swart streke met medium kort woorde om leë sakke te breek en roosters kompakter te maak.
    Gebruik bestaande try_fill_black_regions maar met gefilterde kandidate.
    """
    candidates = [{"answer": normalize_answer(e.get("answers") or e.get("answer","")), "clue": e.get("clues") or e.get("clue","")}
                  for e in remaining_entries]
    candidates = [e for e in candidates if 3 <= len(e["answer"]) <= 7]
    added = try_fill_black_regions(grid, candidates, max_attempts=30)
    return added[:max_added]


# ----------------------------
# Flask routes
# ----------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    global CLUESET
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Lees CSV
    content = io.TextIOWrapper(file.stream, encoding="utf-8", errors="strict")
    reader = csv.DictReader(content)

    # Laat aliase toe vir kolomname
    def pick(name_list):
        for n in name_list:
            if n in reader.fieldnames:
                return n
        return None

    clue_col = pick(["clues", "Clues", "leidraad", "Leidraad"])
    ans_col  = pick(["answers", "Answers", "antwoord", "Antwoord"])
    diff_col = pick(["difficulty", "Difficulty", "moeilikheid", "Moeilikheid"])

    if not clue_col or not ans_col:
        return jsonify({
            "error": "CSV kolomme nie gevind nie. Soek 'clues' of 'Leidraad' en 'answers' of 'Antwoord'.",
            "found": reader.fieldnames
        }), 400

    # Skoonmaak, normaliseer en dedupe
    seen = set()
    cleaned = []
    skipped = []

    for row in reader:
        clue = (row.get(clue_col) or "").strip()
        raw  = (row.get(ans_col) or "").strip()
        # Gebruik bestaande normaliseringsfunksie
        try:
            norm = normalize_answer(raw)
        except Exception:
            norm = "".join(ch for ch in raw.upper().strip() if ch.isalnum())

        if len(norm) < 2:
            skipped.append({"clue": clue, "answer": raw, "reason": "answer too short after normalization"})
            continue
        if norm in seen:
            skipped.append({"clue": clue, "answer": raw, "reason": "duplicate answer (normalized)"})
            continue

        seen.add(norm)
        diff_raw = (row.get(diff_col) or "").strip().lower() if diff_col else ""
        diff_bin = "hard" if diff_raw == "hard" else "easy"
        cleaned.append({"clues": clue, "answers": raw, "difficulty": diff_bin})

    CLUESET = cleaned
    return jsonify({
        "ok": True,
        "count": len(cleaned),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:50]
    })




@app.route("/generate", methods=["POST"])
def generate():
    global CLUESET
    if not CLUESET:
        return jsonify({"error": "Upload a CSV first"}), 400

    data = request.get_json(force=True)
    rows = int(data.get("rows", 15))
    cols = int(data.get("cols", 15))
    cell_size = int(data.get("cell_size", 32))
    # Laat “35%” of “0.35” toe
    black_ratio = _as_ratio(data.get("black_ratio", 0.35), 0.35)
    seed = data.get("seed", None)
    show_solution = bool(data.get("show_solution", False))

    # Simmetrie uit UI
    symmetry_type = data.get("symmetry_type", "none")
    strict_symmetry = _as_bool(data.get("strict_symmetry"), False)

    # NUUT: multistart en parallel beheer
    multistart_runs = max(1, int(data.get("multistart_runs", 8)))
    multistart_workers = int(data.get("multistart_workers", 0))  # 0 beteken Auto


    # Generate unique task ID
    task_id = str(uuid.uuid4())

    def _score_layout(grid, used_words_list) -> float:
        """Hoër beter. Digte invul, minder eilande, gebalanseerde woordlengtes."""
        white = grid.white_count()
        comps = grid_component_count(grid)
        dist = analyze_word_distribution([w.answer for w in used_words_list]) if used_words_list else {"short":0,"medium":0,"long":0}
        balance = 1.0 - abs(0.4 - dist.get("short",0)) - abs(0.4 - dist.get("medium",0)) - abs(0.2 - dist.get("long",0))
        balance = max(0.0, balance)
        return white + 60.0*balance - 10.0*(comps - 1)

    def _one_try(run_idx: int):
        rng_seed = seed if seed is not None else random.randint(1, 1_000_000)
        design_box_local = data.get("design_box", {})

        # Bou ’n kandidaat
        g, used, leftovers = generate_crossword(
            CLUESET,
            rows=rows,
            cols=cols,
            target_black_ratio=black_ratio,
            max_attempts=220,
            rng_seed=rng_seed,
            task_id=task_id,
            design_box=design_box_local,
            symmetry_type=symmetry_type,
            strict_symmetry=strict_symmetry
        )


        # Probeer swart streke verder vul
        try:
            extra = try_fill_black_regions(g, CLUESET, max_attempts=120)
            if extra:
                used.extend(extra)
                g.placed.extend(extra)
        except Exception:
            pass

        score = _score_layout(g, used)
        return (g, used, leftovers, score)

    def generate_async():
        try:
            update_progress(task_id, 0, "Exploring candidate layouts...")
            runs = multistart_runs
            time_limit = _as_int(data.get("time_limit_sec"), 0)
            deadline = time.time() + time_limit if time_limit > 0 else None

            auto_workers = os.cpu_count() or 4
            max_workers = multistart_workers if multistart_workers > 0 else min(auto_workers, max(1, runs))
            best = None

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                # Lansseer eers tot workers vol is
                initial = min(max_workers, max(1, runs))
                futures = {ex.submit(_one_try, i): i for i in range(initial)}
                launched = initial
                done = 0

                while futures:
                    try:
                        for fut in as_completed(list(futures.keys()), timeout=0.25):
                            g, used, leftovers, score = fut.result()
                            done += 1
                            if runs > 0:
                                pct = int(5 + (done / max(1, runs)) * 80)
                            else:
                                pct = min(85, 5 + done)
                            update_progress(task_id, pct, f"Evaluated {done} layouts")

                            if not best or score > best[3]:
                                best = (g, used, leftovers, score)

                            del futures[fut]
                    except FuturesTimeout:
                        # Geen future is binne 250ms klaar nie. Dis ok.
                        pass

                    # As daar nog tyd of runs oor is, lanseer nog werk
                    still_time = (deadline is None) or (time.time() < deadline)
                    more_runs = (launched < runs) if runs > 0 else True
                    can_launch = len(futures) < max_workers
                    if still_time and more_runs and can_launch:
                        futures[ex.submit(_one_try, launched)] = launched
                        launched += 1

                    if not still_time and not futures:
                        break
                    
            if not best:
                raise RuntimeError("No layouts finished. Try increasing time_limit_sec or runs.")
            grid, used, leftovers, _ = best

            # Design box is reeds in generate gebruik. Hou konsekwentheid
            if data.get("design_box", {}).get("enabled"):
                box_size = data["design_box"].get("size", 3)
                start_row = (grid.rows - box_size) // 2
                start_col = (grid.cols - box_size) // 2
                for r in range(start_row, start_row + box_size):
                    for c in range(start_col, start_col + box_size):
                        if 0 <= r < grid.rows and 0 <= c < grid.cols:
                            grid.cells[r][c].ch = None

            update_progress(task_id, 95, "Generating SVG...")
            svg = grid.to_svg(cell_size=cell_size, show_solution=show_solution, design_box=data.get("design_box", {}))

            update_progress(task_id, 97, "Building clue reports...")
            text_report, csv_report = build_clue_report(grid, used, leftovers)

            ts = time.strftime("%Y%m%d-%H%M%S")
            base = f"crossword-{rows}x{cols}-{ts}"
            svg_name = f"{base}.svg"
            txt_name = f"{base}-clues.txt"
            csv_name = f"{base}-clues.csv"
            leftovers_name = f"{base}-leftovers.csv"

            with open(os.path.join(OUTPUT_DIR, svg_name), "w", encoding="utf-8") as f:
                f.write(svg)
            with open(os.path.join(OUTPUT_DIR, txt_name), "w", encoding="utf-8") as f:
                f.write(text_report)
            with open(os.path.join(OUTPUT_DIR, csv_name), "w", encoding="utf-8", newline="") as f:
                f.write(csv_report)

            leftovers_csv = io.StringIO()
            leftovers_writer = csv.writer(leftovers_csv)
            leftovers_writer.writerow(["clues", "answers"])
            for leftover in leftovers:
                leftovers_writer.writerow([leftover.get("clue",""), leftover.get("answer","")])
            with open(os.path.join(OUTPUT_DIR, leftovers_name), "w", encoding="utf-8", newline="") as f:
                f.write(leftovers_csv.getvalue())

            numbering = compute_numbering(grid)
            def add_numbers(pw: PlacedWord) -> Dict[str, Any]:
                return {
                    "clue": pw.clue,
                    "answer": pw.answer,
                    "row": pw.row,
                    "col": pw.col,
                    "length": pw.length,
                    "orientation": "Horizontal" if pw.horizontal else "Vertical",
                    "number": numbering.get((pw.row, pw.col))
                }

            result = {
                "svg": svg,
                "files": {
                    "svg": f"/download/{svg_name}",
                    "clues_txt": f"/download/{txt_name}",
                    "clues_csv": f"/download/{csv_name}",
                    "leftovers_csv": f"/download/{leftovers_name}"
                },
                "used": [add_numbers(pw) for pw in used],
                "leftovers": leftovers
            }

            update_progress(task_id, 100, "Complete!")
            PROGRESS_STORE[task_id]["result"] = result

        except Exception as e:
            PROGRESS_STORE[task_id] = {
                "progress": 0,
                "status": f"Error: {str(e)}",
                "error": True,
                "timestamp": time.time()
            }

    thread = threading.Thread(target=generate_async)
    thread.daemon = True
    thread.start()
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/progress/<task_id>")
def get_progress(task_id):
    """Get progress for a specific task"""
    progress_data = PROGRESS_STORE.get(task_id, {"progress": 0, "status": "Unknown task"})
    return jsonify(progress_data)

@app.route("/bulk_generate", methods=["POST"])
def bulk_generate():
    global CLUESET
    if not CLUESET:
        return jsonify({"error": "Upload a CSV first"}), 400
    
    data = _merge_payload()
    rows = _as_int(data.get("rows"), 15)
    cols = _as_int(data.get("cols"), 15)
    cell_size = _as_int(data.get("cell_size"), 32)
    black_ratio = _as_ratio(data.get("black_ratio") or data.get("black") or data.get("black_percent"), 0.35)
    show_solution = _as_bool(data.get("show_solution"), False)

    # Bulk verwantes: lees dit 1 keer hier, gebruik in closures
    bulk_count = _as_int(data.get("bulk_count"), 0)
    bulk_workers = _as_int(data.get("bulk_workers"), 0)
    sort_by_difficulty = _as_bool(data.get("sort_by_difficulty"), False)

    # Fallback vir enige duplikaat-parallel blokke wat 'max_workers' verwag
    try:
        requested_batches_hint = bulk_count if bulk_count > 0 else max(1, len(CLUESET) // 50)
    except Exception:
        requested_batches_hint = 1
    auto_workers = os.cpu_count() or 4
    max_workers = bulk_workers if bulk_workers > 0 else min(auto_workers, max(1, requested_batches_hint))



    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    def bulk_generate_async():
        try:
            update_progress(task_id, 0, "Starting bulk generation...")
            
            # Nuwe vooraf-balansering vir bulk
            generations = []
            generation_num = 1

            # Laat die gebruiker kies hoeveel roosters om vooraf te balanseer.
            # Indien nie gegee nie, skat dit uit die datastel grootte.
            requested_batches = bulk_count if bulk_count > 0 else max(1, len(CLUESET) // 50)


            # Bou vooraf gebalanseerde bondels
            balanced_batches = _make_balanced_batches(CLUESET.copy(), requested_batches)

            spillover: List[Dict[str, str]] = []

            spillover_min = _as_int(data.get("spillover_min"), 18)
            spillover_limit = _as_int(data.get("spillover_limit"), max(20, requested_batches * 2))


            def _split_by_difficulty(items: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
                """Skei in twee emmers: easy en hard. Enigiets nie 'hard' nie word 'easy'."""
                easy, hard = [], []
                for d in items:
                    if str(d.get("difficulty", "")).strip().lower() == "hard":
                        hard.append(d)
                    else:
                        easy.append(d)
                return easy, hard





            # Hardloop eers deur die vooraf-gebalanseerde bondels
                       # Hardloop die vooraf-gebalanseerde bondels PARALLEL
            # Bou een generasie uit ’n gebalanseerde bondel
            def _build_one(idx: int, batch: List[Dict[str, str]]):
                if len(batch) < 20:
                    return {
                        "generation": idx + 1,
                        "used_count": 0,
                        "leftover_count": len(batch),
                        "files": None,
                        "spillover_items": [{"clues": it["clue"], "answers": it["answer"]} for it in batch]
                    }

                seed = random.randint(1, 1_000_000)
                design_box_local = data.get("design_box", {})
                symmetry_type = data.get("symmetry_type", "none")
                strict_symmetry = _as_bool(data.get("strict_symmetry"), False)
                use_backtracking = _as_bool(data.get("use_backtracking", True), True)

                _ret = generate_crossword(
                    batch, rows=rows, cols=cols, target_black_ratio=black_ratio,
                    max_attempts=220 if use_backtracking else 140,
                    rng_seed=seed, task_id=task_id, design_box=design_box_local,
                    symmetry_type=symmetry_type, strict_symmetry=strict_symmetry
                )

                if isinstance(_ret, tuple) and len(_ret) == 2:
                    grid, used = _ret
                    leftovers = []
                else:
                    grid, used, leftovers = _ret

                try:
                    extra = try_fill_black_regions(grid, batch, max_attempts=120)
                    if extra:
                        used.extend(extra)
                        grid.placed.extend(extra)
                except Exception:
                    pass

                ts = time.strftime("%Y%m%d-%H%M%S")
                base = f"bulk-gen{idx+1}-{rows}x{cols}-{ts}"
                svg_name = f"{base}.svg"
                txt_name = f"{base}-clues.txt"
                csv_name = f"{base}-clues.csv"
                leftovers_name = f"{base}-leftovers.csv"

                svg = grid.to_svg(cell_size=cell_size, show_solution=show_solution, design_box=design_box_local)
                text_report, csv_report = build_clue_report(grid, used, leftovers)

                with open(os.path.join(OUTPUT_DIR, svg_name), "w", encoding="utf-8") as f:
                    f.write(svg)
                with open(os.path.join(OUTPUT_DIR, txt_name), "w", encoding="utf-8") as f:
                    f.write(text_report)
                with open(os.path.join(OUTPUT_DIR, csv_name), "w", encoding="utf-8", newline="") as f:
                    f.write(csv_report)

                leftovers_csv = io.StringIO()
                writer = csv.writer(leftovers_csv)
                writer.writerow(["clues", "answers"])
                for lo in leftovers:
                    writer.writerow([lo.get("clue") or lo.get("clues") or "", lo.get("answer") or lo.get("answers") or ""])
                with open(os.path.join(OUTPUT_DIR, leftovers_name), "w", encoding="utf-8", newline="") as f:
                    f.write(leftovers_csv.getvalue())

                return {
                    "generation": idx + 1,
                    "used_count": len(used),
                    "leftover_count": len(leftovers),
                    "svg": f"/download/{svg_name}",
                    "txt": f"/download/{txt_name}",
                    "csv": f"/download/{csv_name}",
                    "leftovers": f"/download/{leftovers_name}",
                    "spillover_items": [{"clues": it.get("clue") or it.get("clues") or "",
                                         "answers": it.get("answer") or it.get("answers") or ""} for it in leftovers]
                }

            # PARALLEL: bou die bondels gelyktydig
            update_progress(task_id, 10, "Building balanced batches in parallel...")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_build_one, i, batch): i for i, batch in enumerate(balanced_batches)}
                generations = []
                for n, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    generations.append(res)
                    update_progress(task_id, min(90, 10 + int(80 * n / max(1, len(balanced_batches)))), f"Completed {n}/{len(balanced_batches)}")

            # Rangskik per generasie-nommer en bou finale UI-struktuur
            # Veilig: hou ints eerste, dan enige nie-int etiket soos "Oorblyfsels"
            generations.sort(key=lambda g: (0, g["generation"]) if isinstance(g.get("generation"), int) else (1, 0))

            # ... plaas nuwe kode na hierdie lyn ...
            generations.sort(key=lambda g: (0, g["generation"]) if isinstance(g.get("generation"), int) else (1, 0))

            # NUUT: Versamel al die oorblyfsels en probeer 'n laaste rooster bou
            final_leftovers = []
            seen_leftovers = set()
            for gen_result in generations:
                # Die 'spillover_items' bevat die oorblyfsels vir elke generasie
                if gen_result.get("spillover_items"):
                    for item in gen_result["spillover_items"]:
                        # Gebruik 'n unieke sleutel om duplikate te verwyder
                        key = (item.get("clues", "").strip(), item.get("answers", "").strip())
                        if key[0] and key[1] and key not in seen_leftovers:
                            final_leftovers.append(item)
                            seen_leftovers.add(key)
            
            # As daar genoeg oorblyfsels is, probeer 'n laaste generasie
            if len(final_leftovers) >= 20: # Arbitrêre drempel, bv. 20
                update_progress(task_id, 95, f"Building final crossword with {len(final_leftovers)} leftovers...")
                try:
                    leftover_result = _build_one(-1, final_leftovers) # Gebruik -1 as 'n spesiale indeks
                    if leftover_result and leftover_result.get("svg"):
                        # Pas die generasienommer en etiket aan
                        leftover_result["generation"] = "Oorblyfsels"
                        generations.append(leftover_result)
                except Exception as e:
                    print(f"⚠️ Kon nie finale oorblyfsel-rooster bou nie: {e}")


            # Rangskik weer vir ingeval die 'Oorblyfsels' een bygevoeg is
            generations.sort(key=lambda x: x["generation"] if isinstance(x["generation"], int) else 999)


            result = {
                "generations": len(generations),
                "files": [
                    {
                        "generation": g["generation"],
                        "used_count": g["used_count"],
                        "leftover_count": g["leftover_count"],
                        "svg": g["svg"],
                        "txt": g["txt"],
                        "csv": g["csv"],
                        "leftovers": g["leftovers"]
                    } for g in generations
                ]
            }
            update_progress(task_id, 100, f"Bulk generation complete! Generated {len(generations)} crosswords.")
            PROGRESS_STORE[task_id]["result"] = result

            def _run_batches_parallel(batches: List[List[Dict[str, str]]], phase_label: str):
                nonlocal generation_num, spillover

                if not batches:
                    return

                # Bepaal aantal werkers vir hierdie fase
                req_workers = bulk_workers
                max_workers = req_workers if req_workers > 0 else min(8, max(1, len(batches)))

                import os
                auto_workers = os.cpu_count() or 4
                max_workers = req_workers if req_workers > 0 else min(auto_workers, max(1, len(batches)))

                update_progress(
                    task_id,
                    len(generations) * 7,
                    f"Starting parallel generation of {len(batches)} {phase_label} batches with {max_workers} workers."
                )

                completed = 0
                total_batches = len(batches)

                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_build_one, i, b): i for i, b in enumerate(batches)}
                    for fut in as_completed(futures):
                        i = futures[fut]
                        try:
                            res = fut.result()

                            # Versamel spillover vir die volgende fases
                            if res.get("spillover_items"):
                                spillover.extend(res["spillover_items"])

                            # Voeg net by as daar werklike lêers is vir hierdie generasie
                            if res.get("files"):
                                generations.append({
                                    "generation": generation_num,
                                    "used_count": res["used_count"],
                                    "leftover_count": res["leftover_count"],
                                    "svg": res["files"]["svg"],
                                    "txt": res["files"]["txt"],
                                    "csv": res["files"]["csv"],
                                    "leftovers": res["files"]["leftovers"]
                                })
                                generation_num += 1

                        except Exception as e:
                            print(f"⚠️ Fout in {phase_label} batch {i+1}: {e}")
                        finally:
                            completed += 1
                            pct = int((len(generations) * 7) + (completed / max(1, total_batches)) * 70)
                            update_progress(task_id, min(pct, 90), f"{phase_label}: {completed}/{total_batches} done. Spillover {len(spillover)}.")

            # Hardloop nou die fases
            if sort_by_difficulty:
                easy_pool, hard_pool = _split_by_difficulty(CLUESET)
                easy_batches = _make_balanced_batches(easy_pool.copy(), requested_batches) if easy_pool else []
                hard_batches = _make_balanced_batches(hard_pool.copy(), requested_batches) if hard_pool else []
                _run_batches_parallel(easy_batches, "easy")
                _run_batches_parallel(hard_batches, "hard")
            else:
                batches = _make_balanced_batches(CLUESET.copy(), requested_batches)
                _run_batches_parallel(batches, "all")


            # Sorteer vir deterministiese volgorde in die API-uitset
            # Veilig vir gemengde tipes
            generations.sort(key=lambda g: (0, g["generation"]) if isinstance(g.get("generation"), int) else (1, 0))

            # Stel die volgende generasie nommer vir die spillover-fase
            generation_num = len(generations) + 1


            # Gebruik die oorblywende spillover om nog roosters te bou totdat te min oor is
            while len(spillover) >= spillover_min and generation_num <= spillover_limit:
                update_progress(task_id, len(generations) * 7, f"Generation {generation_num}: Using spillover {len(spillover)} words.")

                seed = random.randint(1, 1000000)
                design_box = data.get("design_box", {})
                symmetry_type = data.get("symmetry_type", "none")
                strict_symmetry = bool(data.get("strict_symmetry", False))
                use_backtracking = _as_bool(data.get("use_backtracking", True), True)

                # Skud en neem 'n gebalanseerde sny uit spillover
                # Hergebruik die balanseerder om 'n enkele batch te bou
                one_batch_list = _make_balanced_batches(spillover.copy(), 1)[0]

                if use_backtracking:
                    grid, used, leftovers = generate_crossword_with_backtracking(
                        one_batch_list, rows=rows, cols=cols, target_black_ratio=black_ratio,
                        max_attempts=25, rng_seed=seed, design_box=design_box,
                        symmetry_type=symmetry_type, strict_symmetry=strict_symmetry
                    )
                else:
                    grid, used, leftovers = generate_crossword(
                        one_batch_list, rows=rows, cols=cols, target_black_ratio=black_ratio,
                        max_attempts=100, rng_seed=seed, design_box=design_box,
                        symmetry_type=symmetry_type, strict_symmetry=strict_symmetry
                    )

                svg = grid.to_svg(cell_size=cell_size, show_solution=show_solution, design_box=design_box)
                text_report, csv_report = build_clue_report(grid, used, leftovers)

                ts = time.strftime("%Y%m%d-%H%M%S")
                base = f"bulk-gen{generation_num}-{rows}x{cols}-{ts}"
                svg_name = f"{base}.svg"
                txt_name = f"{base}-clues.txt"
                csv_name = f"{base}-clues.csv"
                leftovers_name = f"{base}-leftovers.csv"

                with open(os.path.join(OUTPUT_DIR, svg_name), "w", encoding="utf-8") as f:
                    f.write(svg)
                with open(os.path.join(OUTPUT_DIR, txt_name), "w", encoding="utf-8") as f:
                    f.write(text_report)
                with open(os.path.join(OUTPUT_DIR, csv_name), "w", encoding="utf-8", newline="") as f:
                    f.write(csv_report)

                leftovers_csv = io.StringIO()
                leftovers_writer = csv.writer(leftovers_csv)
                leftovers_writer.writerow(["clues", "answers"])
                for leftover in leftovers:
                    leftovers_writer.writerow([leftover["clue"], leftover["answer"]])
                with open(os.path.join(OUTPUT_DIR, leftovers_name), "w", encoding="utf-8", newline="") as f:
                    f.write(leftovers_csv.getvalue())

                generations.append({
                    "generation": generation_num,
                    "used_count": len(used),
                    "leftover_count": len(leftovers),
                    "svg": f"/download/{svg_name}",
                    "txt": f"/download/{txt_name}",
                    "csv": f"/download/{csv_name}",
                    "leftovers": f"/download/{leftovers_name}"
                })

                # Herbou spillover uit nuwe leftovers
                spillover = [{"clues": item["clue"], "answers": item["answer"]} for item in leftovers]
                generation_num += 1

            # Finale resultate
            result = {
                "generations": len(generations),
                "files": generations
            }
            update_progress(task_id, 100, f"Bulk generation complete! Generated {len(generations)} crosswords.")
            PROGRESS_STORE[task_id]["result"] = result
            
        except Exception as e:
            PROGRESS_STORE[task_id] = {
                "progress": 0,
                "status": f"Error: {str(e)}",
                "error": True,
                "timestamp": time.time()
            }
    
    # Start background generation
    thread = threading.Thread(target=bulk_generate_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id, "status": "started"})

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    # Skakel die reloader en debugger af om Windows sokket-foute te keer
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
