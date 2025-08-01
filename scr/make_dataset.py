import re, json, random, hashlib, textwrap
from pathlib import Path
from docx import Document
from tqdm import tqdm
import pandas as pd

CATALOG_PATH   = "Каталог.docx"
OUT_DIR        = Path("dataset")
MIN_TOKENS_ANS = 60          
LANG           = "ru"        


doc = Document(CATALOG_PATH)
full_text = "\n".join(p.text for p in doc.paragraphs)


blocks = re.split(r"(?=№\s*\d+\s*-)", full_text)
blocks = [b.strip() for b in blocks if b.strip()]

def extract_fields(block:str) -> dict|None:
    """
    Возвращает dict c ключами:
      num, name, brand, family, top, heart, base, season, description
    Если критичных полей нет – возвращает None.
    """
    lines = block.splitlines()
    header = lines[0]
    m = re.match(r"№\s*(\d+)\s*-\s*([^–]+?)\s*–\s*(.+)", header)
    if not m:
        return None
    num, name, descr = m.groups()
    brand = name.split()[0]   

    def g(pattern):           #
        r = re.search(pattern, block, flags=re.I)
        return r.group(1).strip() if r else None

    return {
        "num":       num,
        "name":      name.strip(),
        "brand":     brand,
        "family":    g(r"Тип аромату:\s*([^\n]+)"),
        "top":       g(r"Верхні ноти:\s*([^\n]+)"),
        "heart":     g(r"Ноти серця:\s*([^\n]+)"),
        "base":      g(r"Базові ноти:\s*([^\n]+)"),
        "season":    g(r"Сезонність:\s*([^\n]+)"),
        "description": descr + " " + " ".join(lines[1:]),
        "raw":       block,
    }

parsed = [extract_fields(b) for b in blocks]
parsed = [p for p in parsed if p and len(p["description"].split()) >= MIN_TOKENS_ANS]

# ---------- 3. build synthetic Q/A pairs ---------- #
QA = []
TEMPLATES = [
    ("Порекомендуй аромат на {season} для дневного ношения", 
     lambda p: f"{p['name']} отлично подойдёт: верхние ноты {p['top']}, сердце {p['heart']}, база {p['base']}."),
    ("Какой аромат с нотами {top} и {base} ты посоветуешь?", 
     lambda p: f"Обратите внимание на {p['name']} – он сочетает {p['top']} в старте и {p['base']} в базе, создавая запоминающийся шлейф."),
    ("Опиши аромат {name} тремя предложениями", 
     lambda p: textwrap.shorten(p["description"], 350, placeholder='…')),
]

for p in parsed:
    for tpl_q, tpl_a in TEMPLATES:
        # skip if нужных полей нет
        if "{season}" in tpl_q and not p["season"]: 
            continue
        q = tpl_q.format(**p)
        a = tpl_a(p)
        QA.append({
          "messages":[
            {"role":"user","content":q},
            {"role":"assistant","content":a}
          ],
          "meta": {"num":p["num"]}
        })


random.shuffle(QA)
split = int(len(QA)*0.9)
train, eval = QA[:split], QA[split:]

OUT_DIR.mkdir(exist_ok=True)
def dump(ds,name):
    with open(OUT_DIR/f"{name}.jsonl","w",encoding="utf-8") as f:
        for row in ds:
            json.dump(row,f,ensure_ascii=False)
            f.write("\n")
dump(train,"train")
dump(eval,"eval")


stat = pd.DataFrame(parsed)
stat.to_markdown(OUT_DIR/"stats.md", index=False)
print(f"✅  Done!  Pairs: {len(QA)},  perfumes parsed: {len(parsed)}")
