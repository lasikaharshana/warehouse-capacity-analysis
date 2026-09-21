"""
Synthetic data generator for the warehouse capacity analysis project.

Creates three files in data/:
  sku_master.csv          600 SKUs with carton dimensions and supplier pallet builds
  inventory_snapshot.csv  batch-level stock on hand, by stock type
  order_lines.csv         four months of outbound order lines

All company, customer and product-brand details are fictional.
Product descriptions use generic (non-proprietary) medicine names.
Fixed random seed, so every run produces identical data.
"""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

# =============================================================================
# 1. PRODUCT FORMS
#    Each form drives realistic carton size, density, storage and cost.
#    upc = units per case. L/W/H in cm. dens = kg per m3. cost = NZD per unit.
# =============================================================================
FORMS = {
    "Tablets":     dict(n=260, cat="Pharma",   upc=[60, 100, 120, 180, 240], L=(30, 60), W=(25, 45), H=(15, 35), dens=(150, 300), cost=(2, 25),    store={"Below 25C": 1.0}),
    "Capsules":    dict(n=70,  cat="Pharma",   upc=[60, 100, 120, 200],      L=(30, 55), W=(25, 40), H=(15, 30), dens=(140, 280), cost=(3, 30),    store={"Below 25C": 1.0}),
    "Syrups":      dict(n=40,  cat="Pharma",   upc=[24, 36, 48, 60],         L=(30, 45), W=(20, 35), H=(15, 25), dens=(550, 800), cost=(3, 12),    store={"Below 25C": 1.0}),
    "IV Fluids":   dict(n=25,  cat="Pharma",   upc=[10, 12, 20, 24],         L=(40, 60), W=(30, 40), H=(20, 30), dens=(850, 1000), cost=(1.5, 6),  store={"Below 25C": 1.0}),
    "Injections":  dict(n=40,  cat="Pharma",   upc=[50, 100, 200],           L=(25, 45), W=(20, 35), H=(12, 25), dens=(120, 250), cost=(3, 40),    store={"Below 25C": 0.7, "2-8C": 0.3}),
    "Cold Chain":  dict(n=20,  cat="Pharma",   upc=[10, 20, 50],             L=(20, 40), W=(15, 30), H=(10, 25), dens=(150, 300), cost=(20, 120),  store={"2-8C": 1.0}),
    "Topicals":    dict(n=30,  cat="Pharma",   upc=[48, 72, 96, 144],        L=(30, 45), W=(20, 35), H=(15, 25), dens=(300, 500), cost=(3, 15),    store={"Below 25C": 1.0}),
    "Eye Drops":   dict(n=20,  cat="Pharma",   upc=[100, 200, 300],          L=(25, 40), W=(20, 30), H=(10, 20), dens=(150, 300), cost=(4, 30),    store={"Below 25C": 1.0}),
    "Inhalers":    dict(n=12,  cat="Pharma",   upc=[50, 100],                L=(35, 55), W=(25, 35), H=(15, 25), dens=(150, 280), cost=(8, 35),    store={"Below 25C": 1.0}),
    "Nutrition":   dict(n=15,  cat="Pharma",   upc=[6, 12, 24],              L=(30, 45), W=(25, 35), H=(20, 30), dens=(350, 550), cost=(15, 45),   store={"Below 25C": 0.6, "Ambient": 0.4}),
    "Surgical":    dict(n=48,  cat="Surgical", upc=[10, 20, 50, 100],        L=(40, 65), W=(30, 45), H=(25, 45), dens=(60, 180),  cost=(0.5, 8),   store={"Ambient": 1.0}),
    "Diagnostics": dict(n=20,  cat="Surgical", upc=[1, 5, 10],               L=(20, 35), W=(15, 25), H=(10, 20), dens=(100, 250), cost=(40, 300),  store={"2-8C": 0.7, "Ambient": 0.3}),
}

# =============================================================================
# 2. PRODUCT NAMES  (generic molecule x strength x pack)
# =============================================================================
def names(molecules, strengths, template, packs):
    return [template.format(m=m, s=s, p=p) for (m, s), p in product(
        [(m, s) for m, ss in zip(molecules, strengths) for s in ss], packs)]

CATALOGUE = {
    "Tablets": names(
        ["Atorvastatin", "Metformin", "Amlodipine", "Losartan", "Pantoprazole", "Clopidogrel",
         "Rosuvastatin", "Paracetamol", "Ibuprofen", "Azithromycin", "Ciprofloxacin", "Cetirizine",
         "Levothyroxine", "Gliclazide", "Bisoprolol", "Telmisartan", "Esomeprazole", "Montelukast",
         "Sertraline", "Escitalopram", "Prednisolone", "Diclofenac", "Simvastatin", "Warfarin",
         "Allopurinol", "Furosemide", "Metoprolol", "Carvedilol", "Glimepiride", "Loratadine"],
        [["10mg", "20mg", "40mg"], ["500mg", "850mg"], ["5mg", "10mg"], ["50mg", "100mg"],
         ["20mg", "40mg"], ["75mg"], ["5mg", "10mg", "20mg"], ["500mg"], ["200mg", "400mg"],
         ["250mg", "500mg"], ["250mg", "500mg"], ["10mg"], ["50mcg", "100mcg"], ["80mg"],
         ["2.5mg", "5mg"], ["40mg", "80mg"], ["20mg", "40mg"], ["10mg"], ["50mg", "100mg"],
         ["10mg", "20mg"], ["5mg", "25mg"], ["50mg"], ["20mg", "40mg"], ["1mg", "5mg"],
         ["100mg", "300mg"], ["40mg"], ["50mg", "100mg"], ["6.25mg", "25mg"], ["2mg", "4mg"], ["10mg"]],
        "{m} {s} Tablets {p}", ["10x10", "3x10", "30s", "2x14", "100s"]),
    "Capsules": names(
        ["Amoxicillin", "Omeprazole", "Fluoxetine", "Doxycycline", "Pregabalin", "Cefalexin",
         "Gabapentin", "Tamsulosin", "Itraconazole", "Vitamin D3", "Fish Oil Omega-3", "Iron Folic"],
        [["250mg", "500mg"], ["20mg", "40mg"], ["20mg"], ["100mg"], ["75mg", "150mg"], ["250mg", "500mg"],
         ["300mg"], ["400mcg"], ["100mg"], ["1000IU"], ["1000mg"], ["150mg"]],
        "{m} {s} Capsules {p}", ["10x10", "3x10", "30s", "60s", "2x14"]),
    "Syrups": names(
        ["Paracetamol 120mg/5ml Syrup", "Amoxicillin 125mg/5ml Dry Syrup", "Cetirizine 5mg/5ml Syrup",
         "Lactulose 3.3g/5ml Solution", "Antacid Oral Suspension", "Salbutamol 2mg/5ml Syrup",
         "Ibuprofen 100mg/5ml Suspension", "Cough Expectorant Syrup", "Multivitamin Syrup",
         "Iron Tonic Syrup", "Azithromycin 200mg/5ml Suspension", "Oral Rehydration Solution",
         "Loratadine 5mg/5ml Syrup", "Zinc Sulfate Syrup"],
        [[""]] * 14, "{m} {p}", ["60ml", "100ml", "200ml"]),
    "IV Fluids": names(
        ["Sodium Chloride 0.9%", "Dextrose 5%", "Compound Sodium Lactate", "Glucose 4% + NaCl 0.18%",
         "Dextrose 10%", "Sodium Chloride 0.45%"],
        [[""]] * 6, "{m} IV {p}", ["250ml Bag", "500ml Bag", "1000ml Bag", "500ml Bottle", "1000ml Bottle"]),
    "Injections": names(
        ["Ceftriaxone 1g Vial", "Ondansetron 4mg/2ml Ampoule", "Heparin 5000IU/ml 5ml Vial",
         "Enoxaparin 40mg Prefilled Syringe", "Dexamethasone 4mg/ml Ampoule", "Tramadol 100mg/2ml Ampoule",
         "Metoclopramide 10mg/2ml Ampoule", "Furosemide 20mg/2ml Ampoule", "Pantoprazole 40mg Vial",
         "Cefuroxime 750mg Vial", "Lidocaine 2% 5ml Ampoule", "Hydrocortisone 100mg Vial",
         "Amikacin 500mg/2ml Vial", "Ketorolac 30mg/ml Ampoule"],
        [[""]] * 14, "{m} {p}", ["1s", "5s", "10s"]),
    "Cold Chain": names(
        ["Insulin Glargine 100IU/ml", "Insulin Aspart 100IU/ml", "Insulin Isophane 100IU/ml",
         "Insulin Lispro 100IU/ml", "Insulin Biphasic 30/70 100IU/ml"],
        [[""]] * 5, "{m} {p}", ["5x3ml Pen", "10ml Vial", "5x3ml Cartridge", "1x3ml Pen"]) + [
        "Hepatitis B Vaccine 10mcg Vial", "Tetanus Toxoid Vaccine 0.5ml", "Influenza Vaccine 0.5ml PFS",
        "Rabies Vaccine 1ml Vial", "Erythropoietin 4000IU PFS", "Filgrastim 300mcg PFS"],
    "Topicals": names(
        ["Hydrocortisone 1% Cream", "Clotrimazole 1% Cream", "Fusidic Acid 2% Cream",
         "Betamethasone 0.1% Ointment", "Mupirocin 2% Ointment", "Diclofenac 1% Gel",
         "Silver Sulfadiazine 1% Cream", "Aciclovir 5% Cream", "Calamine Lotion",
         "Miconazole 2% Cream", "Salicylic Acid Ointment", "Emollient Cream"],
        [[""]] * 12, "{m} {p}", ["15g", "30g", "50g"]),
    "Eye Drops": names(
        ["Timolol 0.5%", "Latanoprost 0.005%", "Chloramphenicol 0.5%", "Ciprofloxacin 0.3%",
         "Artificial Tears", "Prednisolone Acetate 1%", "Brimonidine 0.2%", "Tobramycin 0.3%",
         "Olopatadine 0.1%", "Moxifloxacin 0.5%", "Ketorolac 0.5%"],
        [[""]] * 11, "{m} Eye Drops {p}", ["5ml", "10ml"]),
    "Inhalers": names(
        ["Salbutamol 100mcg", "Budesonide 200mcg", "Budesonide/Formoterol 200/6mcg",
         "Fluticasone 125mcg", "Tiotropium 18mcg", "Ipratropium 20mcg", "Beclometasone 100mcg"],
        [[""]] * 7, "{m} Inhaler {p}", ["120 Dose", "200 Dose"]),
    "Nutrition": names(
        ["Adult Complete Nutrition Powder", "Diabetic Nutrition Powder", "Infant Formula Stage 1",
         "Infant Formula Stage 2", "Growing-Up Milk Stage 3", "High Protein Supplement",
         "Senior Nutrition Powder", "Weight Gain Powder"],
        [[""]] * 8, "{m} {p}", ["400g Tin", "800g Tin"]),
    "Surgical": names(
        ["Nitrile Examination Gloves", "Latex Surgical Gloves", "Syringe Luer Lock", "IV Cannula",
         "Gauze Swab", "Crepe Bandage", "Surgical Mask 3-Ply", "Blood Administration Set",
         "Urinary Catheter", "Adhesive Wound Dressing", "Hypodermic Needle", "Infusion Set"],
        [["Size S", "Size M", "Size L"], ["Size 7", "Size 7.5", "Size 8"], ["3ml", "5ml", "10ml"],
         ["18G", "20G", "22G"], ["7.5x7.5cm", "10x10cm"], ["7.5cm", "10cm"], ["Standard"],
         ["20 drops/ml"], ["14Fr", "16Fr"], ["6x7cm", "9x10cm"], ["21G", "23G"], ["Standard"]],
        "{m} {s} {p}", ["x50", "x100"]),
    "Diagnostics": names(
        ["Blood Glucose Test Strips", "HbA1c Reagent Kit", "Troponin Rapid Test",
         "HIV 1/2 Rapid Test", "Hepatitis B Surface Antigen Test", "Pregnancy Test Strips",
         "Malaria Antigen Test", "CRP Reagent Kit", "Urine Test Strips 10-Parameter",
         "Dengue NS1 Rapid Test", "Cholesterol Reagent Kit", "TSH Reagent Kit"],
        [[""]] * 12, "{m} {p}", ["25 Tests", "100 Tests"]),
}

# Fictional principals and the product forms each supplies (weights = relative size)
PRINCIPALS = {
    "Corvane Pharma":          ({"Tablets": 5, "Capsules": 3}, "Tablets"),
    "Halvora Healthcare":      ({"Tablets": 4, "Syrups": 3, "Topicals": 1}, "Tablets"),
    "Marwick Pharmaceuticals": ({"Tablets": 3, "Inhalers": 5, "Capsules": 1}, "Tablets"),
    "Venrick Laboratories":    ({"Tablets": 3, "Capsules": 3}, "Tablets"),
    "Arcadine Pharma":         ({"Tablets": 2, "Eye Drops": 1}, "Tablets"),
    "Dunmore Therapeutics":    ({"Injections": 4, "Tablets": 1}, "Injections"),
    "Tessaline Biologics":     ({"Cold Chain": 5, "Injections": 2}, "Cold Chain"),
    "Clearford Infusions":     ({"IV Fluids": 5, "Injections": 1}, "IV Fluids"),
    "Pellaro Eye & Skin":      ({"Eye Drops": 4, "Topicals": 3}, "Eye Drops"),
    "Solandra Health":         ({"Syrups": 2, "Topicals": 2, "Capsules": 1}, "Syrups"),
    "Kinloch Nutrition":       ({"Nutrition": 5}, "Nutrition"),
    "Aldermoor Medical":       ({"Surgical": 5}, "Surgical"),
    "Quintessa Diagnostics":   ({"Diagnostics": 5}, "Diagnostics"),
}

# =============================================================================
# 3. BUILD THE SKU MASTER
# =============================================================================
rows = []
prefix = {"Pharma": "PH", "Surgical": "SG"}
counter = {"PH": 10001, "SG": 20001}

for form, f in FORMS.items():
    picks = rng.choice(CATALOGUE[form], f["n"], replace=False)
    suppliers = [p for p, (forms, _) in PRINCIPALS.items() if form in forms]
    weights = np.array([PRINCIPALS[p][0][form] for p in suppliers], float)
    for desc in picks:
        pre = prefix[f["cat"]]
        L, W, H = (round(rng.uniform(*f[k]), 1) for k in ("L", "W", "H"))
        cbm = L * W * H / 1e6
        rows.append(dict(
            sku=f"{pre}{counter[pre]}",
            description=desc,
            principal=rng.choice(suppliers, p=weights / weights.sum()),
            category=f["cat"],
            product_group=form,
            storage_condition=rng.choice(list(f["store"]), p=list(f["store"].values())),
            units_per_case=int(rng.choice(f["upc"])),
            case_length_cm=L, case_width_cm=W, case_height_cm=H,
            case_cbm=round(cbm, 5),
            case_weight_kg=round(cbm * rng.uniform(*f["dens"]), 2),
            unit_cost_nzd=round(rng.uniform(*f["cost"]), 2),
        ))
        counter[pre] += 1

sku = pd.DataFrame(rows)

# Supplier pallet build (Ti = cases per layer, Hi = layers) on a 120 x 100 cm pallet
def max_ti(l, w):
    return int(max((120 // l) * (100 // w), (120 // w) * (100 // l)))

ti_max = np.array([max_ti(l, w) for l, w in zip(sku.case_length_cm, sku.case_width_cm)])
conservative = (rng.random(len(sku)) < 0.25) & (ti_max > 3)     # some suppliers under-fill layers
sku["ti_cases_per_layer"] = np.where(conservative, np.maximum(ti_max - rng.integers(1, 3, len(sku)), 1), ti_max)

target_height = rng.choice([70, 90, 110, 130, 150], len(sku), p=[0.15, 0.2, 0.25, 0.2, 0.2])
hi = np.maximum(target_height // sku.case_height_cm, 1).astype(int)
max_hi_by_weight = np.maximum(1000 // (sku.ti_cases_per_layer * sku.case_weight_kg), 1).astype(int)
sku["hi_layers"] = np.minimum(hi, max_hi_by_weight)

# =============================================================================
# 4. PLANT MASTER DATA ERRORS (about 5%) - the analysis must find these
# =============================================================================
err = rng.choice(len(sku), 30, replace=False)
sku.loc[err[0:6], "case_height_cm"] = 0                                 # missing dimension
sku.loc[err[6:12], "case_cbm"] = (sku.loc[err[6:12], "case_cbm"] * 10).round(5)   # decimal slip
sku.loc[err[12:18], "case_weight_kg"] = 0                               # missing weight
sku.loc[err[18:24], "ti_cases_per_layer"] = ti_max[err[18:24]] + rng.integers(3, 8, 6)  # Ti won't physically fit
sku.loc[err[24:30], "hi_layers"] = 0                                    # missing Hi

sku.to_csv(OUT / "sku_master.csv", index=False)

# =============================================================================
# 5. ORDER LINES  (1 Nov 2025 - 28 Feb 2026, weekdays only)
# =============================================================================
days = pd.bdate_range("2025-11-01", "2026-02-28")
form_boost = {"Tablets": 1.3, "IV Fluids": 2.0, "Syrups": 1.1, "Surgical": 1.2, "Diagnostics": 0.6, "Cold Chain": 0.8}
popularity = rng.lognormal(0, 1.5, len(sku)) * sku.product_group.map(form_boost).fillna(1.0).values
p_sku = popularity / popularity.sum()

N_LINES = 52_000
channel = rng.choice(["Pharmacy", "Hospital", "Distributor"], N_LINES, p=[0.60, 0.25, 0.15])
cust_count = {"Pharmacy": 260, "Hospital": 38, "Distributor": 12}
cust_code = {"Pharmacy": "PHM", "Hospital": "HSP", "Distributor": "DST"}
cust = [f"{cust_code[c]}-{rng.integers(1, cust_count[c] + 1):04d}" for c in channel]

idx = rng.choice(len(sku), N_LINES, p=p_sku)
upc = sku.units_per_case.values[idx]
qty = np.select(
    [channel == "Pharmacy", channel == "Hospital"],
    [rng.integers(2, 13, N_LINES), rng.integers(10, 120, N_LINES)],
    default=np.where(rng.random(N_LINES) < 0.6, upc * rng.integers(1, 8, N_LINES), rng.integers(20, 300, N_LINES)),
)

orders = pd.DataFrame({
    "order_date": rng.choice(days, N_LINES),
    "customer_id": cust,
    "channel": channel,
    "sku": sku.sku.values[idx],
    "qty_units": qty,
}).sort_values(["order_date", "customer_id"]).reset_index(drop=True)
orders.insert(0, "order_no", orders.groupby(["order_date", "customer_id"]).ngroup() + 500001)
orders.to_csv(OUT / "order_lines.csv", index=False)

# =============================================================================
# 6. INVENTORY SNAPSHOT  (as at 28 Feb 2026, batch level)
# =============================================================================
SNAP = pd.Timestamp("2026-02-28")
monthly = orders.groupby("sku").qty_units.sum() / 4
monthly = sku.sku.map(monthly).fillna(0).values

cover = np.clip(rng.lognormal(np.log(5), 0.6, len(sku)), 0.5, 18)       # months of stock cover
cover = np.where(monthly < np.percentile(monthly, 30), cover * 2, cover)  # slow movers overstocked
on_hand = np.where(monthly > 0, monthly * cover, sku.units_per_case.values * rng.integers(1, 4, len(sku)))

inv_rows = []
for i, s in sku.iterrows():
    total = int(on_hand[i])
    n_batch = min(1 + rng.poisson(0.9), 6)
    split = rng.dirichlet(np.ones(n_batch))
    for b, share in enumerate(split):
        q = int(total * share)
        if q == 0:
            continue
        inv_rows.append(dict(sku=s.sku, batch=f"{s.sku[-3:]}{rng.integers(10000, 99999)}",
                             expiry_date=SNAP + pd.Timedelta(days=int(rng.integers(90, 1080))),
                             stock_type="Saleable", qty_units=q))
    # Non-saleable stock that still occupies space
    for stype, prob, frac, exp in [("FOC", 0.08, 0.05, (90, 720)), ("Sample", 0.06, 0.02, (90, 720)),
                                   ("Expired", 0.06, 0.08, (-365, -1)), ("Damaged", 0.03, 0.03, (30, 720))]:
        if rng.random() < prob:
            inv_rows.append(dict(sku=s.sku, batch=f"{s.sku[-3:]}{rng.integers(10000, 99999)}",
                                 expiry_date=SNAP + pd.Timedelta(days=int(rng.integers(*exp))),
                                 stock_type=stype, qty_units=max(int(total * frac), 1)))

inv = pd.DataFrame(inv_rows)
inv["snapshot_date"] = SNAP.date()
inv.to_csv(OUT / "inventory_snapshot.csv", index=False)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"sku_master.csv          {len(sku):>7,} SKUs | {sku.principal.nunique()} principals")
print(f"order_lines.csv         {len(orders):>7,} lines | {orders.order_no.nunique():,} orders | {orders.customer_id.nunique()} customers")
print(f"inventory_snapshot.csv  {len(inv):>7,} batch rows | {inv.sku.nunique()} SKUs in stock")
