# COFGen — COF Property Prediction & Design Tool

**Predict, simulate, and reverse-engineer Covalent Organic Frameworks from building blocks alone — or directly from monomer CIF files.**

COFGen computes pore geometry, simulates powder X-ray diffraction, predicts stacking preferences, estimates gas adsorption, band gap, mechanical properties, stability, and generates synthesis conditions — in seconds, no DFT required.

---

## What's new in this release

- **`from-monomers` command** — predict full COF properties from two monomer CIF files (node + linker). Automatically detects functional groups, infers linkage, estimates arm lengths, and runs the full property pipeline.
- **Imide linkage** — complete support for polyimide COFs (PI-COFs). Includes L2_PMDA, L2_NTCDA, L2_BTDA linkers calibrated from PI-COF-1/2/3 literature unit cells.
- **Porphyrin bug fixes** — stacking probabilities, layer spacing, thermal stability, and CO₂/N₂ selectivity all now vary correctly between different porphyrin + linker combinations.
- **24 smoke tests** — all passing, including new imide and porphyrin distinctness tests.

---

## Quickstart

```bash
git clone https://github.com/your-username/cofgen.git
cd cofgen
python3 -m venv cofgen_env && source cofgen_env/bin/activate
pip install -r requirements.txt
python3 cofgen_tool.py predict --node T3_BENZ --linker L2_PYRN --linkage imine
```

---

## Commands

```bash
# Full property prediction from BB names
python3 cofgen_tool.py predict --node T3_BENZ --linker L2_PYRN --linkage imine

# Predict directly from two monomer CIF files (NEW)
python3 cofgen_tool.py from-monomers \
    --node-cif   TAPB_amine.cif \
    --linker-cif PDA_aldehyde.cif

# Imide PI-COF from monomer CIFs
python3 cofgen_tool.py from-monomers \
    --node-cif   triamine_node.cif \
    --linker-cif PMDA_dianhydride.cif

# From an existing COF CIF
python3 cofgen_tool.py predict --cif my_structure.cif --json results.json

# Simulate PXRD and save plot
python3 cofgen_tool.py pxrd --node T3_BENZ --linker L2_PYRN --linkage imine --plot

# Stacking geometry analysis
python3 cofgen_tool.py stacking --cif my_structure.cif

# Synthesis conditions + DoE matrix
python3 cofgen_tool.py synthesis --node T3_BENZ --linker L2_PYRN --linkage imine

# Diagnose a failed synthesis
python3 cofgen_tool.py synthesis --node T3_BENZ --linker L2_PYRN --linkage imine \
    --observation "gel-like precipitate, no filterable powder"

# Reverse-engineer building blocks from a CIF
python3 cofgen_tool.py reverse --cif unknown_cof.cif

# Generate candidates targeting properties
python3 cofgen_tool.py generate --linkage imine --co2 3.5 --bet 2000 -n 30

# Imide COF prediction
python3 cofgen_tool.py predict --node T3_BENZ --linker L2_PMDA --linkage imide

# List all building blocks
python3 cofgen_tool.py list-bbs
```

Add `--json output.json` to any command to save structured results.

---

## from-monomers: predict from monomer CIF files

The `from-monomers` command takes two CIF files — one for the node building block (amine, boronic acid) and one for the linker (aldehyde, dianhydride, diol) — and predicts the full COF property profile.

```bash
python3 cofgen_tool.py from-monomers \
    --node-cif   your_node.cif \
    --linker-cif your_linker.cif \
    --json       result.json
```

**What it does automatically:**
1. Parses each monomer CIF and builds the atom bond graph
2. Detects functional groups: NH₂, CHO, anhydride, B(OH)₂, NHNH₂, CN
3. Infers the linkage chemistry from the functional group pair
4. Determines connectivity (ditopic L2, tritopic T3, tetratopic S4) from the number of reactive sites
5. Estimates arm length from molecular geometry
6. Matches to the nearest known building block for best-estimate arm length
7. Generates a synthetic COF CIF and runs the full property pipeline

**Tips for best results:**
- CIF files must contain complete atomic positions (including H atoms if possible)
- Files from the CCDC, IsoStar, or any crystallography database work well
- For ambiguous cases, override topology with `--topology hcb` or `--topology sql`
- For imide COFs, ensure the dianhydride CIF contains the full anhydride geometry

**Supported functional group combinations:**

| Node func | Linker func | Linkage |
|---|---|---|
| -NH₂ | -CHO | imine |
| -NH₂ | -anhydride | imide |
| -NH₂ | -β-diketone | beta-ketoenamine |
| -NHNH₂ | -CHO | hydrazone |
| -B(OH)₂ | -B(OH)₂ | boroxine |
| -B(OH)₂ | -diol | boronate ester |
| -CN | -CN | triazine |

---

## What gets predicted

| Property | Output |
|---|---|
| Pore geometry | Void fraction, BET surface area, PLD, LCD, density, cell parameter |
| PXRD | Full simulated pattern with (hkl) peak list (Cu Kα, Cromer-Mann scattering factors) |
| Stacking | AA/AB/ABC probability (BB-specific, calibrated per node type), layer spacing |
| Gas adsorption | CO₂ @ 298K/1bar · CH₄ @ 298K/65bar · H₂ @ 77K/100bar · N₂ @ 298K/1bar · CO₂/N₂ selectivity |
| Electronics | Band gap (eV) with semiconductor classification |
| Mechanical | In-plane and out-of-plane Young's modulus, bulk modulus, interlayer shear |
| Stability | Thermal decomposition temp, water/acid/base stability, hydrolysis half-life |
| Synthesis | Solvent, catalyst, temperature, time, 10-experiment DoE matrix |
| Failure diagnosis | Class A–D per Chen et al. JACS 2026 taxonomy |

---

## Building blocks

### Tritopic nodes (hcb, kgm, hxl topologies)
`T3_BENZ` · `T3_TRIZ` · `T3_TPM` · `T3_TPA` · `T3_TRIF` · `T3_INTZ`

### Tetratopic nodes (sql topology)
`S4_BENZ` · `S4_PORPH` · `S4_PHTH`

### Linkers — imine/hydrazone
`L2_BENZ` · `L2_NAPH` · `L2_BIPH` · `L2_TPHN` · `L2_ANTR` · `L2_PYRN` · `L2_AZBN` · `L2_ETBE` · `L2_STIL` · `L2_BTTA`

### Linkers — imide (dianhydrides)
`L2_PMDA` · `L2_NTCDA` · `L2_BTDA`

### Linkage types
`imine` · `boronate_ester` · `boroxine` · `beta_ketoenamine` · `hydrazone` · `triazine` · `imide`

---

## Repository structure

```
cofgen/
├── cofgen_tool.py                        ← CLI entry point (all commands)
├── requirements.txt
├── analysis/
│   ├── pxrd_simulator.py                 ← structure factor PXRD, stacking analysis
│   ├── property_predictor.py             ← band gap, adsorption, mechanical, stability
│   ├── monomer_reverse_engineer.py       ← CIF → building block identity + SMILES
│   └── monomer_cif_to_cof.py            ← monomer CIF → full COF prediction (NEW)
├── data/
│   ├── crystal_graph.py                  ← CIF parser → CrystalGraph
│   ├── property_labels.py               ← geometric BET, void fraction, PLD
│   └── synthetic_cif_generator.py       ← reticular geometry CIF builder
├── decoder/
│   ├── reticular_decoder.py             ← BB library (now includes imide linkers)
│   └── validity_checker.py              ← synth score (linker-aware penalties)
├── models/
│   ├── synthesis_condition_predictor.py ← synthesis RAG (imide conditions added)
│   └── synthesizability.py
├── scripts/                             ← training pipeline (GPU)
├── evaluation/
└── tests/
    └── smoke_test.py                    ← 24 tests, all pass without GPU
```

---

## Training the generative model (GPU required)

```bash
pip install torch torch-geometric pymatgen pyCOFBuilder rdkit-pypi

python scripts/build_dataset.py  --raw_dir data/raw/ --out_dir data/processed/ --n_jobs 32
python scripts/compute_properties.py --data data/processed/ --n_jobs 32
python scripts/train_encoder.py      --data data/processed/ --epochs 100 --device cuda
python scripts/train_flowmatch.py    --encoder_ckpt checkpoints/encoder/best.pt
python scripts/train_adapter.py      --property co2_uptake_298k_1bar

python scripts/generate.py \
  --checkpoint checkpoints/flowmatch_base/best.pt \
  --adapter_co2 checkpoints/adapter_co2/best.pt \
  --co2_target 4.0 --bet_target 2500 --linkage imine --n 500
```

---

## Citation

```bibtex
@software{cofgen2026,
  title = {COFGen: Property Prediction and Generative Design for Covalent Organic Frameworks},
  year  = {2026},
  url   = {https://github.com/your-username/cofgen}
}
```

Synthesis conditions use the workflow from:
> Chen, L. et al. *Chemist-Guided Human–AI Workflow for Covalent Organic Framework Synthesis.* J. Am. Chem. Soc. **148**, 7440–7449 (2026). https://doi.org/10.1021/jacs.5c20068

---

## License

MIT — see [LICENSE](LICENSE)
